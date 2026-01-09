import os
import webbrowser
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEnginePage


class WebProfileSingleton:
    _profile = None

    @classmethod
    def get_profile(cls):
        if cls._profile is None:
            cache_dir = os.path.abspath("~/.cache/mai_bias_app")
            os.makedirs(cache_dir, exist_ok=True)

            cls._profile = QWebEngineProfile("mai-bias")
            cls._profile.setCachePath(cache_dir)
            cls._profile.setPersistentStoragePath(cache_dir)

        return cls._profile


class ExternalLinkPage(QWebEnginePage):
    def __init__(self, parent=None):
        super().__init__(WebProfileSingleton.get_profile(), parent)

    def acceptNavigationRequest(self, url, _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            webbrowser.open(url.toString())  # open externally
            return False
        return super().acceptNavigationRequest(url, _type, isMainFrame)
