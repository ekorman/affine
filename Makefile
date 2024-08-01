weaviate-docker:
	docker run -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:1.26.1

qdrant-docker:
	docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

coverage:
	coverage run --source=affine -m pytest