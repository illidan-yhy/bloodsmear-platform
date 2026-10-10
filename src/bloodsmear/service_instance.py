"""Kernel-owned single-instance locks and early HTTP port reservation."""
import errno
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import socket


class ServiceLock:
    def __init__(self, database: Path, *, port: int | None = None):
        self.database = Path(database).resolve()
        self.path = self.database.with_name(self.database.name + ".service.lock")
        self.port = port
        self.stream = None
        self.acquired = False

    def acquire(self) -> bool:
        if self.acquired:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600), "r+b")
        try:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
        self.acquired = True
        try:
            # Byte zero is the Windows lock region. Metadata is readable from byte one.
            self.stream.seek(1)
            self.stream.write(json.dumps({"service": "bloodsmear-platform", "pid": os.getpid(), "port": self.port}).encode("utf-8"))
            self.stream.truncate()
            self.stream.flush()
        except BaseException:
            self.release()
            raise
        return True

    def owner_url(self) -> str | None:
        try:
            with self.path.open("rb") as stream:
                stream.seek(1)
                owner = json.loads(stream.read(4096))
            if not isinstance(owner, dict):
                return None
            port = owner.get("port")
            if owner.get("service") == "bloodsmear-platform" and type(port) is int and 1 <= port <= 65535:
                return f"http://127.0.0.1:{port}"
        except (OSError, ValueError):
            pass
        return None

    def release(self) -> None:
        if self.stream is None:
            return
        try:
            if self.acquired:
                self.stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        finally:
            self.acquired = False
            self.stream.close()
            self.stream = None


def reserve_port(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    listener = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(2048)
        listener.set_inheritable(True)
        return listener
    except BaseException:
        listener.close()
        raise


def platform_is_running(port: int) -> bool:
    connection = HTTPConnection("127.0.0.1", port, timeout=0.5)
    try:
        connection.request("GET", "/health/live")
        response = connection.getresponse()
        payload = json.loads(response.read(4096))
        if response.status != 200 or not isinstance(payload, dict) or payload.get("status") != "alive":
            return False
        connection.request("GET", "/api/v1/models/current")
        response = connection.getresponse()
        payload = json.loads(response.read(8192))
        return response.status == 200 and isinstance(payload, dict) and {"name", "version", "sha256", "classes"}.issubset(payload)
    except (OSError, ValueError, TypeError, HTTPException):
        return False
    finally:
        connection.close()
