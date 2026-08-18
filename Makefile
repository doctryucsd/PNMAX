# Makefile — reviewer entry point for the PNMAX artifact.
#
# Wraps ./setup.sh and the per-figure experiments/<name>/run.sh scripts so the
# whole evaluation runs as `make setup`, `make fig10`, `make all`. See README.md.
#
#   make setup          # one-time: env + simulators + PPA DB
#   make fig10           # reproduce Figure 10 (full scale)
#   make fig10 SMOKE=1   # minutes-scale end-to-end check
#   make fig10 ARGS="--workers 8 --seed 7"   # forward flags to run.sh
#   make setup ARGS=--without-cinm          # skip the long CINM/LLVM build
#
# fig10 runs first in `make all`: it builds the mapping pool that
# fig11/fig12/fig15/fig16 reuse. Each run.sh is standalone and resumable, so
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
	@echo '  make fig8 ... fig17 reproduce one figure (run fig10 first: builds the shared pool;'
	@echo '                      no fig9/fig13 - those paper figures have no experiment)'
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
# Figures (fig8 == fig08; paper Figs. 9/13 are analysis-only, no experiment)
#####################################################################
.PHONY: fig8 fig08 fig9 fig10 fig11 fig12 fig13 fig14 fig15 fig16 fig17

fig8 fig08:
	@$(call RUN,fig08_tab4_validation)

fig10:
	@$(call RUN,fig10_pareto)

fig11:
	@$(call RUN,fig11_streaming)

fig12:
	@$(call RUN,fig12_breakdown)

fig14:
	@$(call RUN,fig14_geometry)

fig15:
	@$(call RUN,fig15_buffer)

fig16:
	@$(call RUN,fig16_interconnect)

fig17:
	@$(call RUN,fig17_reduction)

fig9 fig13:
	@echo 'Paper Figs. 9 and 13 are analysis figures with no artifact experiment;'
	@echo 'this artifact reproduces fig8, fig10-fig12 and fig14-fig17. See README.md.'

#####################################################################
# AttAcc area analysis (validation section)
#####################################################################
.PHONY: attacc_area
attacc_area:
	@$(call RUN,attacc_area)

#####################################################################
# Aggregates — fig10 before its pool-derived figures (fig11/12/15/16)
#####################################################################
.PHONY: all figures
all figures:
	@$(call RUN,fig08_tab4_validation)
	@$(call RUN,fig10_pareto)
	@$(call RUN,fig11_streaming)
	@$(call RUN,fig12_breakdown)
	@$(call RUN,fig15_buffer)
	@$(call RUN,fig16_interconnect)
	@$(call RUN,fig14_geometry)
	@$(call RUN,fig17_reduction)
	@$(call RUN,attacc_area)

# Sample test: exercise every pipeline end-to-end at smoke scale (~12 min).
.PHONY: smoke test
smoke test:
	@for e in fig08_tab4_validation fig10_pareto fig11_streaming fig12_breakdown \
	          fig15_buffer fig16_interconnect fig14_geometry fig17_reduction attacc_area; do \
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
