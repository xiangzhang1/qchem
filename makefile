clean:
	find data/*pickle.20* -mindepth 1 -mtime +7 -delete
push:
	git add -A ; git commit -am 'automated push' ; git push
gui:
	git pull ; ./gui/gui.py
package:
	tar --exclude gui.py -zcf gui.tar.gz gui
