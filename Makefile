NETWORK = sentiment-net
VOLUME = sentiment-logs
API_IMAGE = sentiment-api
MONITORING_IMAGE = sentiment-monitoring
API_CONTAINER = sentiment-api
MONITORING_CONTAINER = sentiment-monitoring

.PHONY: build run clean

build:
	docker build -t $(API_IMAGE) ./api
	docker build -t $(MONITORING_IMAGE) ./monitoring

run:
	docker network inspect $(NETWORK) >/dev/null 2>&1 || docker network create $(NETWORK)
	docker volume inspect $(VOLUME) >/dev/null 2>&1 || docker volume create $(VOLUME)
	docker run -d --name $(API_CONTAINER) \
		--network $(NETWORK) \
		-v $(VOLUME):/logs \
		-p 8000:8000 \
		$(API_IMAGE)
	docker run -d --name $(MONITORING_CONTAINER) \
		--network $(NETWORK) \
		-v $(VOLUME):/logs \
		-p 8501:8501 \
		$(MONITORING_IMAGE)
	@echo "API:        http://localhost:8000"
	@echo "Monitoring: http://localhost:8501"

clean:
	-docker rm -f $(API_CONTAINER) $(MONITORING_CONTAINER)
	-docker network rm $(NETWORK)
	-docker volume rm $(VOLUME)
