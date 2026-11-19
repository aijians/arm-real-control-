"""Register an upright camera frame against fixed parts of the same guide rail."""

import cv2
import numpy as np


class RailRegistrationError(ValueError):
    """The frame cannot be measured in the reference coordinate system."""


class RailRegistration:
    """Translation-only ECC over separate fixed-structure regions.

    The shared transform must explain each patch. Sobel texture and local
    correlation checks reject flat, obscured or geometrically changed patches.
    Coordinates are on the upright reference PNG at its native resolution.
    """

    def __init__(self, reference, rois, *, max_shift_px=8.0, min_correlation=0.80):
        self.size = reference.shape[1::-1]
        width, height = self.size
        if (not isinstance(rois, (list, tuple)) or len(rois) < 3
                or any(not isinstance(r, (list, tuple)) or len(r) != 4
                       or any(type(v) is not int for v in r) for r in rois)):
            raise RailRegistrationError("registration_rois requires at least three [x1,y1,x2,y2] patches")
        pad = int(np.ceil(max_shift_px)) + 5
        if any(x2 - x1 < 12 or y2 - y1 < 12 or x1 < pad or y1 < pad
               or x2 + pad > width or y2 + pad > height for x1, y1, x2, y2 in rois):
            raise RailRegistrationError("registration ROI is too small or too close to the frame edge")
        self.max_shift_px, self.min_correlation = max_shift_px, min_correlation
        self.x0 = min(r[0] for r in rois) - pad
        self.y0 = min(r[1] for r in rois) - pad
        self.x1 = max(r[2] for r in rois) + pad
        self.y1 = max(r[3] for r in rois) + pad
        self.slices = [(slice(y1 - self.y0, y2 - self.y0), slice(x1 - self.x0, x2 - self.x0))
                       for x1, y1, x2, y2 in rois]
        self.reference = self._texture(reference)
        self.mask = np.zeros(self.reference.shape, np.uint8)
        for region in self.slices:
            self.mask[region] = 255
            if self.reference[region].std() < 3:
                raise RailRegistrationError("registration reference contains an untextured ROI")
        gx = cv2.Sobel(self.reference, cv2.CV_32F, 1, 0)[self.mask != 0]
        gy = cv2.Sobel(self.reference, cv2.CV_32F, 0, 1)[self.mask != 0]
        eigenvalues = np.linalg.eigvalsh(np.cov([gx, gy]))
        if eigenvalues[0] < 1.0 or eigenvalues[0] < 0.02 * eigenvalues[1]:
            raise RailRegistrationError("registration reference does not constrain both translation axes")

    def _texture(self, image):
        patch = image[self.y0:self.y1, self.x0:self.x1]
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray - cv2.GaussianBlur(gray, (0, 0), 3.0)

    def estimate_shift(self, upright):
        """Return the reference-to-current offset without resampling the frame."""
        if upright.shape[1::-1] != self.size:
            raise RailRegistrationError("registration image size differs from reference")
        moving = self._texture(upright)
        warp = np.eye(2, 3, dtype=np.float32)
        radius = int(np.ceil(self.max_shift_px))
        scores = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        for ys, xs in self.slices:
            search = moving[ys.start-radius:ys.stop+radius, xs.start-radius:xs.stop+radius]
            scores += cv2.matchTemplate(search, self.reference[ys, xs], cv2.TM_CCOEFF_NORMED)
        dy, dx = np.unravel_index(scores.argmax(), scores.shape)
        warp[:, 2] = dx - radius, dy - radius
        try:
            score, warp = cv2.findTransformECC(
                self.reference, moving, warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 45, 1e-6),
                self.mask, 3,
            )
        except cv2.error as exc:
            raise RailRegistrationError("rail_registration_did_not_converge") from exc
        shift = warp[:, 2]
        if (not np.all(np.isfinite(shift)) or not np.isfinite(score)
                or np.linalg.norm(shift) > self.max_shift_px
                or score < self.min_correlation):
            raise RailRegistrationError("rail_registration_shift_or_score_out_of_range")
        registered = cv2.warpAffine(moving, warp, self.reference.shape[::-1],
                                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        scores = []
        for region in self.slices:
            ref, live = self.reference[region].ravel(), registered[region].ravel()
            ref = ref - ref.mean()
            live = live - live.mean()
            norm = np.linalg.norm(ref) * np.linalg.norm(live)
            scores.append(float(np.clip(np.dot(ref, live) / norm, -1.0, 1.0)) if norm > 0 else 0.0)
        if min(scores) < self.min_correlation:
            raise RailRegistrationError("rail_registration_patch_disagreement")
        return shift.copy(), warp, {"shift_from_reference_px": shift.tolist(),
                             "ecc": float(score), "roi_correlations": scores}

    def align(self, upright):
        _, warp, info = self.estimate_shift(upright)
        aligned = cv2.warpAffine(upright, warp, self.size,
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        return aligned, info
