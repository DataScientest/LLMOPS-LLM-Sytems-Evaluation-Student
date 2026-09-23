build:
	@echo "🏗️  Building Docker images..."
	docker compose build

up:
	@echo "🚀 Starting RAGOPS services..."
	docker compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "✅ RAGOPS is ready!"
	@echo "   Backend: http://localhost:18000"
	@echo "   Health:  http://localhost:18000/health"

# --profile manual also removes the evaluator container (otherwise it keeps a dangling network)
down:
	@echo "⏹️  Stopping RAGOPS services..."
	docker compose --profile manual down

restart:
	@echo "🔄 Restarting RAGOPS services..."
	docker compose --profile manual down
	docker compose up -d

logs:
	@echo "📄 Showing service logs..."
	docker-compose logs -f backend

build-eval:
	cd src/eval && uv lock && cd ../.. && docker compose build evaluator

# --build: `make build` skips the evaluator (manual profile), so rebuild it here from src/eval
run-eval:
	docker compose --profile manual up --build evaluator

links:
	@echo "   Meilisearch: http://localhost:7700"
	@echo "   Backend: http://localhost:18000"
	@echo "   Health:  http://localhost:18000/health"