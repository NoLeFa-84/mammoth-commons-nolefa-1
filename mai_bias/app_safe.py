import os

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
print(
    "You are running mai_bias.app_safe. Module results that require WebGL will not render correctly."
)

import sys
from mai_bias.app import QApplication, MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
