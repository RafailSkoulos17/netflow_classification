CC	=	g++
CFLAGS	=	-O0 -g
#CFLAGS	=	-O2 -g
SOURCES = 	*.cpp
MAIN = main.cpp

#SOURCES := $(filter-out evaluation/lsh.cpp, $(SOURCES))
#LIBS := $(filter-out evaluation/lsh.h, $(LIBS))

SOURCESPYTHON =	apta.cpp dfasat.cpp  refinement.cpp evaluation_factory.cpp random_greedy.cpp  state_merger.cpp parameters.cpp searcher.cpp stream.cpp interactive.cpp
LFLAGS 	= 	-std=c++11 -L/opt/local/lib -I/opt/local/include -I./lib -I. -lm -lpopt -lgsl -lgslcblas
PYTHON_EVAL = evaluation/python.cpp

EVALFILES := $(wildcard evaluation/*.cpp)
EVALOBJS := $(addprefix evaluation/,$(notdir $(EVALFILES:.cpp=.o)))

ifdef WITH_PYTHON
  PYTHON_VERSION=$(shell python3 -c 'import sys; print("".join([str(v) for v in sys.version_info[:2]]))')
  PYTHON_INC=$(shell python3-config --includes)
  PYTHON_LIBS=$(shell python3-config --libs)
  BOOST_LIBS=-lboost_python-py$(PYTHON_VERSION)
else
  EVALFILES := $(filter-out $(PYTHON_EVAL), $(EVALFILES))
  EVALOBJS := $(addprefix evaluation/,$(notdir $(EVALFILES:.cpp=.o)))
endif


OUTDIR ?= .

.PHONY: all clean

all: regen gitversion.cpp flexfringe

regen:
	sh collector.sh

debug:
	$(CC) -g $(SOURCES) -o flexfringe $(LFLAGS) $(LIBS)

main: $(EVALOBJS)
	$(CC) $(CFLAGS) -o $@ $(MAIN) $^ -I./ $(LFLAGS) $(LIBS)

flexfringe: $(EVALOBJS)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $^ -I./ $(LFLAGS) $(LIBS)

evaluation/%.o: evaluation/%.cpp
	$(CC) -fPIC -c -o $@ $< -I./lib $(LFLAGS) $(LIBS) $(PYTHON_INC) $(PYTHON_LIBS) $(BOOST_LIBS)

clean:
	rm -f flexfringe ./evaluation/*.o generated.cpp named_tuple.py *.dot exposed_decl.pypp.txt flexfringe*.so gitversion.cpp

gitversion.cpp: .git/HEAD .git/index
	echo "const char *gitversion = \"$(shell git rev-parse HEAD)\";" > $@

python: $(EVALOBJS) gitversion.cpp
	export CPLUS_INCLUDE_PATH=/usr/include/python3.5
	$(CC) -fPIC -shared $(CFLAGS)  -o flexfringe.lib.so $(SOURCESPYTHON) $^ -I./ $(LFLAGS) $(LIBS) $(PYTHON_LIBS) $(PYTHON_INC)
	python3 generate.py
	g++ -W  -g -Wall -fPIC -shared generated.cpp flexfringe.lib.so -o flexfringe.so $(PYTHON_LIBS) $(PYTHON_INC) $(BOOST_LIBS) $(LFLAGS) -Wl,-rpath,'$$ORIGIN' -L. -l:flexfringe.lib.so
	mv flexfringe.lib.so flexfringe.so $(OUTDIR)
