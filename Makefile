.PHONY: setup data train test dashboard-build up down logs

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

data:
	python -m src.data.generate_synthetic --machines 60 --days 90

train:
	python -m src.models.train_streaming_models

test:
	pytest -q

dashboard-build:
	cd dashboard && npm install && npm run build

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100
