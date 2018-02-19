clean:
	find data/*pickle.20* -mindepth 1 -mtime +7 -delete
push:
	git add -A ; git commit -am 'automated push' ; git push
pull:
	git pull
