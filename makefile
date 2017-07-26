.PHONY : compile, clean, test
compile : *.src.py
	for f in *.src.py ; do sed "s/'''//g" "$$f" > "$${f%.src.py}.py"; done
clean :
	for f in *.src.py ; do rm -rf "$${f%.src.py}.py"; done; rm -rf *.pyc
run :
	make ; flask run
