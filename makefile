clean:
	find data/*pickle.20* -mindepth 1 -mtime +1 -delete
push:
	git add -A ; git commit -am 'automated push' ; git push
cli:
	git pull ; ./cli.py
gui:
	git pull ; ./gui.py
