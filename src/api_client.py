from typing import Any, Dict, List, Optional, Tuple
import requests

from backend.config import BACKEND_URL, SESSION_COOKIE_NAME


class APIClient:
    """HTTP Client connecting Streamlit Frontend to FastAPI Backend."""

    def __init__(self, base_url: str = BACKEND_URL, session_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.session_token = session_token

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
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 201:
                return True, resp.json()
            error_data = resp.json()
            return False, error_data.get("detail", "Registration failed.")
        except Exception as e:
            return False, f"Could not reach backend service: {e}"

    def login(self, email: str, password: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/auth/login"
        payload = {"email": email, "password": password}
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.session_token = data.get("token")
                return True, data
            error_data = resp.json()
            return False, error_data.get("detail", "Login failed.")
        except Exception as e:
            return False, f"Could not reach backend service: {e}"

    def logout(self) -> Tuple[bool, str]:
        url = f"{self.base_url}/api/auth/logout"
        try:
            resp = requests.post(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=8)
            self.session_token = None
            if resp.status_code == 200:
                return True, "Logged out successfully."
            return False, "Logout request failed."
        except Exception as e:
            self.session_token = None
            return False, str(e)

    def get_me(self) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/auth/me"
        try:
            resp = requests.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=8)
            if resp.status_code == 200:
                return True, resp.json()
            return False, resp.json().get("detail", "Unauthorized")
        except Exception as e:
            return False, str(e)

    def list_conversations(self) -> Tuple[bool, List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/conversations"
        try:
            resp = requests.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            return False, []
        except Exception:
            return False, []

    def create_conversation(self, title: str = "New Technical Inquiry") -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/conversations"
        try:
            resp = requests.post(url, json={"title": title}, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            if resp.status_code == 201:
                return True, resp.json()
            return False, resp.json().get("detail", "Failed to create conversation.")
        except Exception as e:
            return False, str(e)

    def rename_conversation(self, conv_id: str, new_title: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/conversations/{conv_id}"
        try:
            resp = requests.patch(url, json={"title": new_title}, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            return False, resp.json().get("detail", "Failed to rename conversation.")
        except Exception as e:
            return False, str(e)

    def delete_conversation(self, conv_id: str) -> Tuple[bool, str]:
        url = f"{self.base_url}/api/conversations/{conv_id}"
        try:
            resp = requests.delete(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            if resp.status_code == 200:
                return True, "Conversation deleted."
            return False, resp.json().get("detail", "Failed to delete conversation.")
        except Exception as e:
            return False, str(e)

    def get_messages(self, conv_id: str) -> Tuple[bool, List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/conversations/{conv_id}/messages"
        try:
            resp = requests.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            return False, []
        except Exception:
            return False, []

    def send_message(self, conv_id: str, content: str) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/conversations/{conv_id}/messages"
        try:
            resp = requests.post(url, json={"content": content}, headers=self._get_headers(), cookies=self._get_cookies(), timeout=45)
            if resp.status_code == 200:
                return True, resp.json()
            return False, resp.json().get("detail", "Failed to get response.")
        except Exception as e:
            return False, str(e)

    def admin_list_documents(self) -> Tuple[bool, List[Dict[str, Any]]]:
        url = f"{self.base_url}/api/admin/documents"
        try:
            resp = requests.get(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=10)
            if resp.status_code == 200:
                return True, resp.json()
            return False, []
        except Exception:
            return False, []

    def admin_upload_document(self, filename: str, file_bytes: bytes) -> Tuple[bool, Any]:
        url = f"{self.base_url}/api/admin/documents/upload"
        files = {"file": (filename, file_bytes, "application/pdf")}
        try:
            resp = requests.post(url, files=files, headers=self._get_headers(), cookies=self._get_cookies(), timeout=60)
            if resp.status_code == 201:
                return True, resp.json()
            return False, resp.json().get("detail", "Failed to upload document.")
        except Exception as e:
            return False, str(e)

    def admin_delete_document(self, doc_id: str) -> Tuple[bool, str]:
        url = f"{self.base_url}/api/admin/documents/{doc_id}"
        try:
            resp = requests.delete(url, headers=self._get_headers(), cookies=self._get_cookies(), timeout=15)
            if resp.status_code == 200:
                return True, resp.json().get("message", "Document deleted.")
            return False, resp.json().get("detail", "Failed to delete document.")
        except Exception as e:
            return False, str(e)
