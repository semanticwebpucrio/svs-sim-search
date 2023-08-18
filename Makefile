.PHONY: start stop destroy

start:
	echo "Building project"
	docker-compose build api
	echo "Running multimodal-clustering background"
	echo "Starting Redis"
	docker-compose up -d redis
	echo "Starting API"
	docker-compose up -d api

stop:
	echo "Stopping multimodal-clustering"
	docker-compose down

destroy:
	echo "Stopping multimodal-clustering and removing volumes"
	docker-compose down -v
