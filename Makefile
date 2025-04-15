.PHONY: all lib example clean

CC = gcc

CFLAGS = -Wall -ansi -pedantic -s -lm
INCDIR = include
INCLUDES = -I./$(INCDIR)
SRCDIR = src
BINDIR = bin
CFILES = $(SRCDIR)/cNNFW.c
HFILES = $(INCDIR)/cNNFW.h

LIBNAME = cnnfw

ifeq ($(OS),Windows_NT)
	TARGETS = $(patsubst  apps/%.c,$(BINDIR)/%.exe,$(wildcard apps/*.c))
    EXT = .exe
    LIBEXT = .dll
    RM = if exist $(BINDIR) rd /s /q
    MKDIR = if not exist $(BINDIR) md
    ECHO = echo
    ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
        
    else
        ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
            
        endif
        ifeq ($(PROCESSOR_ARCHITECTURE),x86)
            
        endif
    endif
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        TARGETS = $(patsubst  apps/%.c,$(BINDIR)/%,$(wildcard apps/*.c))
        EXT = 
        LIBEXT = .so
        RM = rm -rfv
        MKDIR = mkdir -pv
        ECHO = echo
    endif
    ifeq ($(UNAME_S),Darwin)
        TARGETS = $(patsubst  apps/%.c,$(BINDIR)/%,$(wildcard apps/*.c))
        EXT = 
        LIBEXT = .dylib
        RM = rm -rfv
        MKDIR = mkdir -pv
        ECHO = echo
    endif
    UNAME_P := $(shell uname -p)
    ifeq ($(UNAME_P),x86_64)
        
    endif
    ifneq ($(filter %86,$(UNAME_P)),)
        
    endif
    ifneq ($(filter arm%,$(UNAME_P)),)
        
    endif
endif

LIB = $(BINDIR)/lib$(LIBNAME)$(LIBEXT)

all:
	@$(ECHO) "* make lib - to build $(patsubst $(BINDIR)/%,%,$(LIB))"
	@$(ECHO) "* make apps - to build $(patsubst $(BINDIR)/%,%,$(LIB)) and $(patsubst $(BINDIR)/%,%,$(TARGETS))"
	@$(ECHO) "* make $(patsubst $(BINDIR)/%$(EXT),run-%,$(TARGETS)) - to run one of the app"
	@$(ECHO) "* make clean - to remove all the binaries"

apps: $(LIB) $(TARGETS)

$(BINDIR)/%$(EXT): apps/%.c $(CFILES) $(HFILES) | $(BINDIR)
	@echo "Building $(@F)"
ifeq ($(UNAME_S),Darwin)
	@$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ -Wl,-rpath,@loader_path -L./$(BINDIR) -l$(LIBNAME)
else
	@$(CC) $(CFLAGS) $(INCLUDES) $< -o $@ -Wl,-rpath=./$(BINDIR)/ -L./$(BINDIR) -l$(LIBNAME)
endif

$(BINDIR):
	@$(MKDIR) $(BINDIR)

lib: $(LIB)

$(LIB): $(CFILES) $(HFILES) | $(BINDIR)
	@echo "Building $(@F)"
ifeq ($(UNAME_S),Darwin)
	@$(CC) $(CFLAGS) -dynamiclib $< -o $@
else
	@$(CC) $(CFLAGS) -shared $(INCLUDES) $< -o $@
endif

run-%:
	@./$(patsubst run-%,$(BINDIR)/%,$@) $(ARGS)

clean:
	@$(RM) $(BINDIR)
