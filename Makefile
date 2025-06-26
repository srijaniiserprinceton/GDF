# Makefile

.PHONY: setup test clean

# Load slepian_dir from the first line of .config
slepian_dir := $(shell head -n 1 .config)

# Main setup task
setup: create_dirs copy_files test

# Step 1: Create necessary directories
create_dirs:
	@echo "📁 Creating directories..."
	mkdir -p Outputs Figures

# Step 2: Copy files (customize paths as needed)
copy_files:
	@echo "📄 Copying Slepian template files..."
	cp ./slepian_templates/*.m $(slepian_dir)/slepian_foxtrot/.

# Step 3: Run Python installation test
test:
	@echo "🐍 Running installation test..."
	python main.py init_gdf_default

# Optional: Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf Outputs/* Figures/*