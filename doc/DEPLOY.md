
```bash 
poetry build
poetry config repositories.testpypi https://test.pypi.org/legacy/
poetry config pypi-token.testpypi pypi-TOKEN_HERE
poetry config --list
poetry publish -r testpypi
```
