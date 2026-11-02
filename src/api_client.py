from typing import Any, Dict, List, Optional, Tuple
import requests

from backend.config import BACKEND_URL, SESSION_COOKIE_NAME


class APIClient:
    """HTTP Client connecting Streamlit Frontend to FastAPI Backend."""

    def __init__(self, base_url: str = BACKEND_URL, session_token: Optional[str] = None):
        raw_url = (base_url or "http://127.0.0.1:8000").strip().rstrip("/")
        # Strip accidental endpoint suffixes if user pasted the full endpoint into BACKEND_URL
        for suffix in ("/api/auth/login", "/api/auth", "/api"):
            if raw_url.endswith(suffix):
                raw_url = raw_url[:-len(suffix)].rstrip("/")

        # Enforce https for onrender.com to prevent 301/302 redirects converting POST to GET (causing HTTP 405)
        if raw_url.startswith("http://") and ("onrender.com" in raw_url or "render.com" in raw_url):
            raw_url = "https://" + raw_url[7:]

        self.base_url = raw_url
        self.session_token = session_token
        self.session = requests.Session()

    def _get_headers(self) -> Dict[str, str]:
        headers = {}
        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"
        return headers

    def _get_cookies(self) -> Dict[str, str]:
        cookies = {}
        if self.session_token:
            cookies[SESSION_COOKIE_NAME] = self.session_token
        return cookies

    def _parse_error(self, resp: requests.Response) -> str:
        """Safely extract error message without throwing JSONDecodeError on HTML responses."""
        try:
            data = resp.json()
            if isinstance(data, dict) and "detail" in data:
                return str(data["detail"])
            return str(data)
        except Exception:
            status = resp.status_code
            if status in (502, 503, 504):
                return (
                    f"Backend service is temporarily unavailable (HTTP {status}). "
                    "If hosted on Render Free Tier, the backend service is likely spinning up from idle "
                    "(please wait 30-50 seconds and try again)."
                )
            elif status == 404:
                return f"Backend endpoint not found (HTTP 404) at '{resp.url}'. Please verify your BACKEND_URL."
            return f"Backend returned HTTP {status}: {resp.text[:120].strip()}"

    def register(self, name: str, email: str, password: str, admin_key: Optional[str] = None) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/auth/register"
        payload = {
            "name": name,
            "email": email,
            "password": password
        }
        if admin_key:
            payload["admin_key"] = admin_key

        try:
            resp = self.session.post(url, json=payload, timeout=20)
            if resp.status_code == 201:
                return True, resp.json()
            return False, self._parse_error(resp)
        except requests.exceptions.ConnectionError:
            return False, f"Could not connect to backend at {self.base_url}. Please check if the backend service is running."
        except Exception as e:
            return False, f"Could not reach backend service: {e}"

    def login(self, email: str, password: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/auth/login"
        payload = {"email": email, "password": password}
        try:
            resp = self.session.post(url, json=payload, timeout=25)
            if resp.status_code == 200:
                data = resp.json()
                self.session_token = data.get("token")
                return True, data
            return False, self._parse_error(resp)
        except requests.exceptions.ConnectionError:
            return False, f"Could not connect to backend at {self.base_url}. Please verify BACKEND_URL."
        except Exception as e:
            return False, f"Could not reach backend service: {e}"

    def logout(self) -> Tuple[bool, str]:
        url = f"{self.base_url}/api/auth/logout"
        try:
            resp = self.session.post(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            self.session_token = None
            if resp.status_code == 200:
                return True, "Logged out successfully."
            return False, self._parse_error(resp)
        except Exception as e:
            self.session_token = None
            return False, str(e)

    def get_me(self) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/auth/me"
        try:
            resp = self.session.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, resp.json()
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)

    def list_conversations(self) -> Tuple[bool, List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/conversations"
        try:
            resp = self.session.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, resp.json()
            return False, []
        except Exception:
            return False, []

    def create_conversation(self, title: str = "New Technical Inquiry") -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/conversations"
        try:
            resp = self.session.post(url, json={"title": title}, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 201:
                return True, resp.json()
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)

    def rename_conversation(self, conv_id: str, new_title: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/conversations/{conv_id}"
        try:
            resp = self.session.patch(url, json={"title": new_title}, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, resp.json()
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)

    def delete_conversation(self, conv_id: str) -> Tuple[bool, str]:
        url = f"{self.base_url}/api/conversations/{conv_id}"
        try:
            resp = self.session.delete(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, "Conversation deleted."
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)

    def get_messages(self, conv_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/conversations/{conv_id}/messages"
        try:
            resp = self.session.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, resp.json()
            return False, []
        except Exception:
            return False, []

    def send_message(self, conv_id: str, content: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/conversations/{conv_id}/messages"
        try:
            resp = self.session.post(url, json={"content": content}, headers=self._get_headers(), cookies=self._get_cookies(), timeout=60)
            if resp.status_code == 200:
                return True, resp.json()
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)

    def admin_list_documents(self) -> Tuple[bool, List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/admin/documents"
        try:
            resp = self.session.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, resp.json()
            return False, []
        except Exception:
            return False, []

    def admin_upload_document(self, filename: str, file_bytes: bytes) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/admin/documents/upload"
        files = {"file": (filename, file_bytes, "application/pdf")}
        try:
            resp = self.session.post(url, files=files, headers=self._get_headers(), cookies=self._get_cookies(), timeout=90)
            if resp.status_code == 201:
                return True, resp.json()
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)

    def admin_delete_document(self, doc_id: str) -> Tuple[bool, str]:
        url = f"{self.base_url}/api/admin/documents/{doc_id}"
        try:
            resp = self.session.delete(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=20)
            if resp.status_code == 200:
                return True, resp.json().get("message", "Document deleted.")
            return False, self._parse_error(resp)
        except Exception as e:
            return False, str(e)
