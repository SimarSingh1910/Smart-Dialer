# Thin wrapper over tasks.py so the project runs the same on every platform.
PY ?= python

.PHONY: up db migrate seed test sim loadtest run

up:        ; $(PY) tasks.py up
db:        ; $(PY) tasks.py db
migrate:   ; $(PY) tasks.py migrate
seed:      ; $(PY) tasks.py seed
test:      ; $(PY) tasks.py test
sim:       ; $(PY) tasks.py sim
loadtest:  ; $(PY) tasks.py loadtest
run:       ; $(PY) tasks.py run
