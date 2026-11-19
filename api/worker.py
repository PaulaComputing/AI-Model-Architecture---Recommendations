import os
import json
import time
import redis
import pika

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")

cache = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def connect_rabbitmq():
    while True:
        try:
            conn = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
            return conn
        except pika.exceptions.AMQPConnectionError:
            print("Esperando a RabbitMQ...")
            time.sleep(3)

def callback(ch, method, properties, body):
    event = json.loads(body.decode("utf-8"))
    user_id = event.get("user_id")
    print(f"[Worker] Evento procesado: {event}")

    # Invalidar caché de recomendaciones del usuario ante nueva interacción
    for k in range(1, 21):
        cache.delete(f"rec:user:{user_id}:k:{k}")

    # Guardar historial reciente en Redis
    cache.lpush(f"history:user:{user_id}", json.dumps(event))
    cache.ltrim(f"history:user:{user_id}", 0, 19)

    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    conn = connect_rabbitmq()
    ch = conn.channel()
    ch.queue_declare(queue="user_events", durable=True)
    ch.basic_qos(prefetch_count=1)
    ch.basic_consume(queue="user_events", on_message_callback=callback)
    print("[Worker] Escuchando eventos en cola 'user_events'...")
    ch.start_consuming()

if __name__ == "__main__":
    main()