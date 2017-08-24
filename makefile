.PHONY : compile, clean, run, test
run : compile
	flask run
test :
	python test.py
