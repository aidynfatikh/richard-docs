import sys
import cv2
import requests

def check_qr_website(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": f"cannot read image: {image_path}"}

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)

    if not data:
        return {"success": False, "error": "QR code not found"}

    url = data.strip()

    if not url.startswith("http"):
        url = "https://" + url

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            )
        }

        r = requests.get(url, headers=headers, timeout=5)

        return {
            "success": True,
            "url": url,
            "alive": r.status_code < 400,
            "status": r.status_code
        }
    except Exception as e:
        return {
            "success": True,
            "url": url,
            "alive": False,
            "error": str(e)
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_qr.py path/to/qr_image.png")
        sys.exit(1)

    print(check_qr_website(sys.argv[1]))
