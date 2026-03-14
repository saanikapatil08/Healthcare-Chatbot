"""
QR Code Generator for Healthcare Chatbot Demo
Run this script after deploying to Streamlit Cloud to generate a QR code
"""

import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask
import sys

def generate_qr_code(url: str, filename: str = "demo_qr.png"):
    """Generate a styled QR code for the demo URL"""
    
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    
    # Add data
    qr.add_data(url)
    qr.make(fit=True)
    
    # Create styled image
    img = qr.make_image(
        image_factory=StyledPilImage,
        module_drawer=RoundedModuleDrawer(),
        color_mask=RadialGradiantColorMask(
            back_color=(255, 255, 255),
            center_color=(31, 119, 180),  # Primary blue
            edge_color=(44, 160, 44)  # Secondary green
        )
    )
    
    # Save image
    img.save(filename)
    print(f"✅ QR code saved to: {filename}")
    print(f"📱 URL encoded: {url}")
    
    return filename


def generate_simple_qr(url: str, filename: str = "demo_qr_simple.png"):
    """Generate a simple QR code (fallback if styled fails)"""
    
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="#1f77b4", back_color="white")
    img.save(filename)
    print(f"✅ Simple QR code saved to: {filename}")
    
    return filename


if __name__ == "__main__":
    # Default URL pattern for Streamlit Cloud
    # Replace YOUR_USERNAME with your actual GitHub username
    # The format is: https://YOUR_APP_NAME.streamlit.app
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Placeholder - update after deployment
        url = "https://healthcare-chatbot-demo.streamlit.app"
        print("⚠️  Using placeholder URL. Run with actual URL:")
        print("   python generate_qr.py https://your-app-name.streamlit.app")
        print()
    
    try:
        generate_qr_code(url)
    except Exception as e:
        print(f"Styled QR failed: {e}")
        print("Generating simple QR code instead...")
        generate_simple_qr(url)
    
    print("\n📋 Instructions:")
    print("1. Add demo_qr.png to your resume/portfolio")
    print("2. Recruiters can scan to view the live demo")
    print("3. Update the URL and regenerate if needed")
