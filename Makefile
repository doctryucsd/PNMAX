# Makefile — reviewer entry point for the PNMAX artifact.
#
# Wraps ./setup.sh and the per-figure experiments/<name>/run.sh scripts so the
# whole evaluation runs as `make setup`, `make fig9`, `make all`. See README.md.
#
#   make setup          # one-time: env + simulators + PPA DB
#   make fig9           # reproduce Figure 9 (full scale)
#   make fig9 SMOKE=1   # minutes-scale end-to-end check
#   make fig9 ARGS="--workers 8 --seed 7"   # forward flags to run.sh
#   make setup ARGS=--without-cinm          # skip the long CINM/LLVM build
#
# fig9 runs first in `make all`: it builds the mapping pool that
# fig10/fig11/fig13/fig14 reuse. Each run.sh is standalone and resumable, so
# any target also rebuilds what it needs on its own.

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

# Flags forwarded verbatim to run.sh / setup.sh. SMOKE=1 prepends --smoke.
ARGS ?=
RUNFLAGS := $(ARGS)
ifeq ($(SMOKE),1)
RUNFLAGS := --smoke $(RUNFLAGS)
endif

# $(call RUN,<experiment-dir>) -> run that experiment with the shared flags.
RUN = experiments/$(1)/run.sh $(RUNFLAGS)

#####################################################################
# Help (default target)
#####################################################################
.PHONY: help
help:
	@echo 'PNMAX artifact — make targets:'
	@echo '  make setup          bootstrap env + build simulators + PPA DB (run once)'
	@echo '  make setup-nocinm   same, skipping the long CINM/LLVM build'
	@echo '  make smoke          ~12 min end-to-end check of every pipeline'
	@echo '  make fig8 ... fig15 reproduce one figure (run fig9 first: builds the shared pool)'
	@echo '  make attacc_area    AttAcc area value (validation section)'
	@echo '  make all            full campaign, ~15 h at 64 workers'
	@echo '  make clean          remove smoke outputs   (clean-all: remove all results)'
	@echo
	@echo 'Flags:  SMOKE=1  quick check     ARGS="--workers N --seed N --dry-run"'

#####################################################################
# Setup
#####################################################################
.PHONY: setup setup-nocinm
setup:
	@./setup.sh $(ARGS)

setup-nocinm:
	@./setup.sh --without-cinm $(ARGS)

#####################################################################
# Figures (fig8 == fig08, fig9 == fig09)
#####################################################################
.PHONY: fig8 fig08 fig9 fig09 fig10 fig11 fig12 fig13 fig14 fig15

fig8 fig08:
	@$(call RUN,fig08_tab4_validation)

fig9 fig09:
	@$(call RUN,fig09_pareto)

fig10:
	@$(call RUN,fig10_streaming)

fig11:
	@$(call RUN,fig11_breakdown)

fig12:
	@$(call RUN,fig12_geometry)

fig13:
	@$(call RUN,fig13_buffer)

fig14:
	@$(call RUN,fig14_interconnect)

fig15:
	@$(call RUN,fig15_reduction)

#####################################################################
# AttAcc area analysis (validation section)
#####################################################################
.PHONY: attacc_area
attacc_area:
	@$(call RUN,attacc_area)

#####################################################################
# Aggregates — fig9 before its pool-derived figures (fig10/11/13/14)
#####################################################################
.PHONY: all figures
all figures:
	@$(call RUN,fig08_tab4_validation)
	@$(call RUN,fig09_pareto)
	@$(call RUN,fig10_streaming)
	@$(call RUN,fig11_breakdown)
	@$(call RUN,fig13_buffer)
	@$(call RUN,fig14_interconnect)
	@$(call RUN,fig12_geometry)
	@$(call RUN,fig15_reduction)
	@$(call RUN,attacc_area)

# Sample test: exercise every pipeline end-to-end at smoke scale (~12 min).
.PHONY: smoke test
smoke test:
	@for e in fig08_tab4_validation fig09_pareto fig10_streaming fig11_breakdown \
	          fig13_buffer fig14_interconnect fig12_geometry fig15_reduction attacc_area; do \
	  echo "==== smoke: $$e ===="; \
	  experiments/$$e/run.sh --smoke $(ARGS) || exit $$?; \
	done

#####################################################################
# Clean (keeps the tracked results/.gitkeep)
#####################################################################
.PHONY: clean clean-all
clean:
	@rm -rf results/smoke

clean-all:
	@find results -mindepth 1 -maxdepth 1 ! -name .gitkeep -exec rm -rf {} +
