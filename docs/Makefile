SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help clean html apidoc

help:
	@echo "Usage: make <target>";
	@echo "  html    - build HTML docs";
	@echo "  apidoc  - run sphinx-apidoc to generate API .rst files from package";
	@echo "  clean   - remove build artifacts";

html:
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR)

apidoc:
	-sphinx-apidoc -o $(SOURCEDIR) ../torchflow

clean:
	rm -rf $(BUILDDIR)/* $(SOURCEDIR)/modules* $(SOURCEDIR)/torchflow*
