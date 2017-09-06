clean:
	find data/ -mindepth 1 -mtime +1 -delete
push:
	git add -A ; git commit -am 'automated push' ; git push
pull:
	git pull ; ./gui.py
