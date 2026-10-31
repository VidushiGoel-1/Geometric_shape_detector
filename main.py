"""
Geometric Shape Detector (Hinglish comments ke saath)
Ye program image ya live video mein shapes dhundta hai, unki type batata hai,
unki boundary (contour) draw karta hai, aur area/perimeter/bounding box nikalta hai.

Kaise chalayein:
    python shape_detector.py --image shapes.png            # ek image par
    python shape_detector.py --image shapes.png --save out.png
    python shape_detector.py --video 0                     # webcam
    python shape_detector.py --video clip.mp4              # video file
    python shape_detector.py --demo                        # test image khud banake chalata hai

Video mode mein keys: 'q' = band karo, 'd' = debug window on/off.
"""

import argparse
import math
import sys

import cv2          # OpenCV: image processing ki main library
import numpy as np  # numpy: arrays aur maths ke liye


# STEP 1: Image preprocessing (image ko saaf karna taaki shapes aasani se milein)
def preprocess(frame, blur_ksize=5):
    """Grayscale -> Blur -> Threshold + Canny edges -> saaf mask.

    Har stage return karte hain taaki hum debug mein dekh sakein.
    """
    # Colour image (BGR) ko black-and-white (grayscale) mein badlo.
    # Colour ki zaroorat nahi, sirf brightness chahiye shape dhundne ke liye.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gaussian blur: image ko thoda dhundhla karta hai taaki chhota-mota noise
    # (dots, grain) hat jaye. Warna noise ko galti se edge samajh lega.
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Thresholding: har pixel ko ya to pura safed (255) ya pura kala (0) bana do.
    # Otsu method khud best threshold value dhundh leta hai.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hume chahiye: shape SAFED aur background KALA.
    # Agar image ke kinare (border) safed hain matlab background safed hai -> ulta (invert) kar do.
    border = np.concatenate([thresh[0, :], thresh[-1, :], thresh[:, 0], thresh[:, -1]])
    if border.mean() > 127:
        thresh = cv2.bitwise_not(thresh)

    # Canny edge detection: jahan brightness achanak badalti hai (shape ki boundary),
    # wahan edge draw karta hai. Kam contrast wali shapes ke liye helpful.
    edges = cv2.Canny(blurred, 50, 150)

    # Threshold aur edges dono ko jod do (OR), phir chhote gaps bhar do
    # taaki har shape ek poora solid blob ban jaye.
    combined = cv2.bitwise_or(thresh, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)  # gaps bharo
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)       # chhote dots hatao
    return gray, blurred, thresh, edges, mask


# STEP 2: Contour detection (shapes ki boundary dhundhna)
def find_shape_contours(mask, min_area=500):
    """Mask se bahari boundaries (contours) nikalo aur chhote noise wale hata do."""
    # RETR_EXTERNAL = sirf bahar ki boundary, andar ke holes ignore.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Jinka area min_area se kam hai woh noise hai, unhe filter out kar do.
    return [c for c in contours if cv2.contourArea(c) >= min_area]


# STEP 3: Polygon approximation aur classification (shape ka naam batana)
def _angles(approx):
    """Polygon ke har corner (vertex) ka angle degrees mein nikalta hai."""
    pts = approx.reshape(-1, 2).astype(float)
    n = len(pts)
    result = []
    for i in range(n):
        # Teen points: pichla, current, aur agla. Current point par angle nikalna hai.
        p_prev, p, p_next = pts[i - 1], pts[i], pts[(i + 1) % n]
        v1, v2 = p_prev - p, p_next - p  # do vectors (do sides)
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            continue
        # Dot product formula se angle: cos(theta) = (v1.v2) / (|v1||v2|)
        cos_a = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        result.append(math.degrees(math.acos(cos_a)))
    return result


def classify_shape(contour, epsilon_factor=0.02):
    """(shape ka naam, approximated polygon) return karta hai.

    approxPolyDP tedhi-medhi contour ko simple polygon bana deta hai.
    Polygon ke corners ginke shape pata chalta hai (3 corners = triangle, etc).
    epsilon = perimeter ka 2%: itna error allowed hai simplify karte time.
    """
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
    sides = len(approx)  # corners ki ginti

    if sides == 3:
        return "Triangle", approx

    if sides == 4:
        # Agar shape andar ki taraf dabi hui hai (non-convex) to simple quadrilateral.
        if not cv2.isContourConvex(approx):
            return "Quadrilateral", approx
        angles = _angles(approx)
        # Rectangle/Square ke saare angles lagbhag 90 degree hote hain.
        all_right = all(abs(a - 90) < 10 for a in angles)
        if all_right:
            # minAreaRect tilted shape ki bhi sahi width/height deta hai.
            (_, _), (w, h), _ = cv2.minAreaRect(approx)
            ratio = max(w, h) / max(min(w, h), 1e-6)
            # Lambai aur chaudai lagbhag barabar -> Square, warna Rectangle.
            return ("Square" if ratio < 1.15 else "Rectangle"), approx
        return "Quadrilateral", approx

    if sides == 5:
        return "Pentagon", approx

    if sides == 6:
        return "Hexagon", approx

    # 6 se zyada corners: check karo kahin circle to nahi.
    # Circularity = 4*pi*area / perimeter^2. Perfect circle ke liye ye 1 hoti hai.
    area = cv2.contourArea(contour)
    circularity = 4 * math.pi * area / (perimeter ** 2) if perimeter else 0
    if sides > 6 and circularity > 0.82:
        return "Circle", approx
    return "Polygon", approx


# STEP 4: Geometric measurements
def measure(contour):
    """Contour ka area, perimeter, bounding box aur center nikalta hai."""
    area = cv2.contourArea(contour)            # shape ke andar kitne pixels
    perimeter = cv2.arcLength(contour, True)   # boundary ki lambai
    x, y, w, h = cv2.boundingRect(contour)     # shape ko gherne wala seedha box
    # Moments se centroid (shape ka center point) nikalte hain.
    m = cv2.moments(contour)
    if m["m00"] != 0:
        cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    else:
        cx, cy = x + w // 2, y + h // 2
    return {"area": area, "perimeter": perimeter, "bbox": (x, y, w, h), "center": (cx, cy)}

# Drawing aur poori pipeline
# Har shape ka apna colour (BGR format mein, RGB nahi!)
COLORS = {
    "Triangle": (0, 0, 255),
    "Square": (0, 200, 0),
    "Rectangle": (255, 128, 0),
    "Quadrilateral": (200, 0, 200),
    "Pentagon": (0, 200, 200),
    "Hexagon": (0, 128, 255),
    "Circle": (255, 0, 0),
    "Polygon": (128, 128, 128),
}


def detect_shapes(frame, min_area=500):
    """Ek frame par poori pipeline chalao. (annotated image, results list, debug images) deta hai."""
    # Step 1: preprocessing
    gray, blurred, thresh, edges, mask = preprocess(frame)
    # Step 2: contours dhundo
    contours = find_shape_contours(mask, min_area)

    output = frame.copy()  # original ko kharab na karo, copy par draw karo
    results = []
    for idx, c in enumerate(contours, start=1):
        name, approx = classify_shape(c)   # Step 3
        m = measure(c)                     # Step 4
        color = COLORS.get(name, (0, 255, 0))
        x, y, w, h = m["bbox"]

        cv2.drawContours(output, [c], -1, color, 2)                          # asli boundary
        cv2.polylines(output, [approx], True, (0, 255, 255), 1)              # simplified polygon (peela)
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 255), 1)    # bounding box
        cv2.circle(output, m["center"], 3, color, -1)                        # center dot

        # Shape ke upar naam aur area/perimeter likho
        label = f"#{idx} {name}"
        cv2.putText(output, label, (x, max(y - 22, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(output, f"A={m['area']:.0f} P={m['perimeter']:.0f}", (x, max(y - 6, 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        results.append({"id": idx, "shape": name, "vertices": len(approx), **m})

    return output, results, {"gray": gray, "blur": blurred, "thresh": thresh,
                             "edges": edges, "mask": mask}


def print_report(results):
    """Terminal mein sundar table print karta hai."""
    print(f"\n{'ID':<4}{'Shape':<15}{'Vertices':<10}{'Area (px²)':<14}"
          f"{'Perimeter (px)':<16}{'BBox (x,y,w,h)'}")
    print("-" * 75)
    for r in results:
        print(f"{r['id']:<4}{r['shape']:<15}{r['vertices']:<10}{r['area']:<14.1f}"
              f"{r['perimeter']:<16.1f}{r['bbox']}")
    print(f"\nTotal shapes detected: {len(results)}")

# Modes (image / video / demo)
def run_image(path, min_area, save_path=None, show=True):
    """Ek image par detection chalao."""
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"Image nahi padh paaye: {path}")
    output, results, _ = detect_shapes(frame, min_area)
    print_report(results)
    if save_path:
        cv2.imwrite(save_path, output)
        print(f"Annotated image save ho gayi: {save_path}")
    if show:
        cv2.imshow("Shape Detector", output)
        cv2.waitKey(0)           # kisi key ka wait karo
        cv2.destroyAllWindows()


def run_video(source, min_area):
    """Webcam ya video file par live detection chalao."""
    # "0" jaise number ka matlab webcam, warna video file ka path.
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"Video source nahi khula: {source}")
    debug = False
    while True:
        ok, frame = cap.read()   # ek frame padho
        if not ok:               # video khatam ya camera band
            break
        output, results, dbg = detect_shapes(frame, min_area)
        cv2.putText(output, f"Shapes: {len(results)}  (q=quit, d=debug)", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Shape Detector", output)
        if debug:
            cv2.imshow("Threshold", dbg["thresh"])
            cv2.imshow("Edges", dbg["edges"])
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):      # q dabaya to band
            break
        if key == ord("d"):      # d dabaya to debug windows on/off
            debug = not debug
            if not debug:
                cv2.destroyWindow("Threshold")
                cv2.destroyWindow("Edges")
    cap.release()
    cv2.destroyAllWindows()


def make_demo_image(path="demo_shapes.png"):
    """Test ke liye ek image banata hai jisme saari shapes hain."""
    img = np.full((500, 800, 3), 255, np.uint8)  # safed canvas
    cv2.fillPoly(img, [np.array([[100, 40], [40, 140], [160, 140]])], (0, 0, 0))              # triangle
    cv2.rectangle(img, (230, 40), (330, 140), (0, 0, 0), -1)                                  # square
    cv2.rectangle(img, (400, 50), (560, 130), (0, 0, 0), -1)                                  # rectangle
    cv2.fillPoly(img, [np.array([[670, 40], [740, 40], [775, 140], [640, 140]])], (0, 0, 0))  # quadrilateral
    pent = [[int(120 + 70 * math.cos(math.radians(-90 + 72 * i))),
             int(300 + 70 * math.sin(math.radians(-90 + 72 * i)))] for i in range(5)]
    cv2.fillPoly(img, [np.array(pent)], (0, 0, 0))                                            # pentagon
    hexa = [[int(300 + 75 * math.cos(math.radians(60 * i))),
             int(300 + 75 * math.sin(math.radians(60 * i)))] for i in range(6)]
    cv2.fillPoly(img, [np.array(hexa)], (0, 0, 0))                                            # hexagon
    cv2.circle(img, (500, 300), 65, (0, 0, 0), -1)                                            # circle
    rot = cv2.boxPoints(((680, 300), (110, 110), 30)).astype(np.int32)
    cv2.fillPoly(img, [rot], (0, 0, 0))                                                       # tilted square
    cv2.imwrite(path, img)
    return path


def main():
    """Command-line arguments padhke sahi mode chalata hai."""
    ap = argparse.ArgumentParser(description="Geometric shape detector")
    ap.add_argument("--image", help="input image ka path")
    ap.add_argument("--video", help="webcam number (jaise 0) ya video file ka path")
    ap.add_argument("--demo", action="store_true", help="demo image banao aur analyse karo")
    ap.add_argument("--min-area", type=float, default=500, help="isse chhote contours ignore (noise filter)")
    ap.add_argument("--save", help="annotated image yahan save karo")
    ap.add_argument("--no-show", action="store_true", help="window mat kholo (headless mode)")
    args = ap.parse_args()

    if args.demo:
        path = make_demo_image()
        run_image(path, args.min_area, args.save or "demo_output.png", show=not args.no_show)
    elif args.image:
        run_image(args.image, args.min_area, args.save, show=not args.no_show)
    elif args.video is not None:
        run_video(args.video, args.min_area)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()