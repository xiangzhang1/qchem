compile: *.src.py
	for filename in *.src.py
	do
	    f=${filename%%.*}
	    cp ${f}.src.py ${f}.py
	    sed -i "s/'''//g" ${f}.py
	done
clean:
	for filename in *.src.py
	do
	    f=${filename%%.*}
		trash *.py
	done
test:
	cp test/* .
	python test.py
	trash test.*
