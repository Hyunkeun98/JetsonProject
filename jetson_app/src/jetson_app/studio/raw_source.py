"""브로커 구독 어댑터: 받은 payload를 가공하지 않고 콜백으로 넘긴다(시험·수집이 같이 쓴다)."""
from __future__ import annotations


class RawSource:
    def __init__(self) -> None:
        self._client = None
        self._connected = False

    def connect(self, host: str, port: int, topics, on_payload) -> None:
        """접속을 시도한다. 브로커에 닿지 못하면 OSError. 접속 뒤에는 백그라운드 스레드가 받는다."""
        import paho.mqtt.client as mqtt

        client = mqtt.Client()

        def on_connect(c, userdata, flags, rc):
            self._connected = rc == 0
            if rc != 0:
                print(f"[studio] MQTT 연결 실패 (rc={rc}) — 브로커 인증/ACL 설정을 확인하세요")
                return
            for topic in topics:
                c.subscribe(topic, qos=1)

        def on_disconnect(c, userdata, rc):
            self._connected = False

        def on_message(c, userdata, msg):
            on_payload(msg.topic, msg.payload)

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        client.connect(host, port, keepalive=60)
        client.loop_start()
        self._client = client

    def disconnect(self) -> None:
        client, self._client = self._client, None
        self._connected = False
        if client is not None:
            client.loop_stop()
            client.disconnect()

    def is_connected(self) -> bool:
        return self._connected
