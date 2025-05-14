FROM python:3.9 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY compute/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -LO https://golang.org/dl/go1.21.0.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz && \
    rm go1.21.0.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

COPY . .
RUN go build -o app ./backend

FROM python:3.9-slim
WORKDIR /app


COPY --from=builder /usr/local/go /usr/local/go
COPY --from=builder /app/app .
COPY --from=builder /app/compute ./compute
COPY --from=builder /root/.cache/pip /root/.cache/pip
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

ENV PATH="/usr/local/go/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.9/site-packages" \
    MPLCONFIGDIR="/tmp/.matplotlib"

EXPOSE 8080
CMD ["./app"]
