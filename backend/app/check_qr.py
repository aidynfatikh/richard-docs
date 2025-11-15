import cv2
from pyzbar.pyzbar import decode
import requests

def check_qr_website(image_path: str):
    # 1. Читаем изображение
    img = cv2.imread(image_path)

    # 2. Декодируем QR
    decoded = decode(img)

    if not decoded:
        return {"success": False, "error": "QR code not found"}

    # Берём первый найденный QR
    qr_data = decoded[0].data.decode("utf-8")

    # 3. Проверяем URL
    try:
        response = requests.get(qr_data, timeout=5)
        is_alive = response.status_code < 400
    except Exception as e:
        return {
            "success": True,
            "url": qr_data,
            "alive": False,
            "error": str(e)
        }

    return {
        "success": True,
        "url": qr_data,
        "alive": is_alive,
        "status": response.status_code
    }

print(check_qr_website("qr.svg"))