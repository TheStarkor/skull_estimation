# Getting Started
- Docker 버전
    ```
    $ docker build -t skull_estimation .
    $ docker run -d -p 80:5000 skull_estimation
    ```

- Flask 버전
    pytorch 설치 환경이어야 함
    ```
    $ pip install flask flask_cors
    $ python app.py
    ```

## 시작 전에 참고하기

모델 가중치 파일 불러온 후 `backend/app.py` 77번째에 파일명 변경해주세요 맞춰서
