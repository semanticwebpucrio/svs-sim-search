#!/usr/bin/env bash

TRY_LOOP="20"

wait_for_port() {
  local name="$1" host="$2" port="$3"
  local j=0
  while ! nc -z "$host" "$port" >/dev/null 2>&1 < /dev/null; do
    j=$((j+1))
    if [ $j -ge $TRY_LOOP ]; then
      echo >&2 "$(date) - $host:$port still not reachable, giving up"
      exit 1
    fi
    echo "$(date) - waiting for $name $host $port... $j/$TRY_LOOP"
    sleep 5
  done
}

# wait_for_port "Redis" "$REDIS_HOST" "$REDIS_PORT"

case "$1" in
    api)
        exec python app/main.py
        ;;
    txt_encoder)
        exec python app/services/text.py "$QUEUE_ID"
        ;;
    img_encoder)
        exec python app/services/image.py "$QUEUE_ID"
        ;;
    *)
        exec "$@"
        ;;
esac
