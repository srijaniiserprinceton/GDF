# Makefile

.PHONY: setup setup_new_gdfdir test clean

# Load slepian_dir from the first line of .config
slepian_dir := $(shell head -n 1 .config)

# Main setup task for the GDF repository
setup: create_dirs create_slepian_directories copy_slepian_template_files

# Setup task for each new gdf run directory
setup_new_gdfdir: create_dirs

# Step 1a: Create necessary directories
create_dirs:
	@echo "📁 Creating directories..."
	@mkdir -p Outputs
	@mkdir -p Figures/mcmc_dists_polcap
	@mkdir -p Figures/span_rec_polcap
	@mkdir -p Figures/super_res_polcap
	@mkdir -p Figures/kneeL_polcap
	@mkdir -p Figures/cartesian_slepians
	@mkdir -p Figures/super_res_cartesian
	@mkdir -p Figures/super_res_hybrid

# Step 1b: Copying the slightly modified Slepian files
create_slepian_directories:
	@mkdir -p $(slepian_dir)/IFILES
	@mkdir -p $(slepian_dir)/IFILES/LEGENDRE
	@mkdir -p $(slepian_dir)/IFILES/SDWCAP

# Step 2: Copy files (customize paths as needed)
copy_slepian_template_files:
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