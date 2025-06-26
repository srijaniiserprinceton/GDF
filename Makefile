# Makefile

.PHONY: setup test clean

# Load slepian_dir from the first line of .config
slepian_dir := $(shell head -n 1 .config)

# Main setup task
setup: create_dirs copy_files

# Step 1: Create necessary directories
create_dirs:
	@echo "📁 Creating directories..."
	@mkdir -p Outputs
	@mkdir -p Figures/mcmc_dists
	@mkdir -p Figures/span_rec_contour
	@mkdir -p Figures/super_res
	@mkdir -p Figures/kneeL
	@mkdir -p $(slepian_dir)/IFILES
	@mkdir -p $(slepian_dir)/IFILES/LEGENDRE
	@mkdir -p $(slepian_dir)/IFILES/SDWCAP

# Step 2: Copy files (customize paths as needed)
copy_files:
	@echo "📄 Copying Slepian template files..."
	@cp ./slepian_templates/*.m $(slepian_dir)/slepian_foxtrot/.

# Step 3: Run Python installation test
testrun:
	@echo "🐍 Running installation test..."
	python main.py init_gdf_default

# Optional: Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -rf Outputs/* Figures/*