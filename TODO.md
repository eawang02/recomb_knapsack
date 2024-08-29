# TODO for `recomb_knapsack`


- [ ] Fill out the README file
    - [ ] Description of the project
    - [ ] How to install
    - [ ] How to run with some small examples
    - [ ] Explain a little of the theory of what is going on with some pictures

- [ ] **Add tests**
    - [ ] Integrate tests with pytest
    - [ ] Include pytest-cov coverage report

- [ ] Add the TEX files to the `docs` folder so that they can be edited on the fly
    - [ ] Make sure to add auxiliary files to the .gitignore

- [X] Make doc strings adhere to the Google-style python docstrings. Note:
    the docstings in `utils` are close but not quite there. Example:

```
def multiply_numbers(a, b):
    """
    Multiplies two numbers and returns the result.

    Args:
        a (int): The first number.
        b (int): The second number.

    Returns:
        int: The product of a and b.
    """
    return a * b
```

- [X] Remove extra comments and extraneous lines across the package
