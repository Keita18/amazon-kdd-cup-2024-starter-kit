## Adding your runtime

This repository is a valid submission (and submission structure). 
You can simply add your dependencies on top of this repository.

Few of the most common ways are as follows:

* `requirements.txt` -- The `pip3` packages used by your inference code. As you add new pip3 packages to your inference procedure either manually add them to `requirements.txt` or if your software runtime is simple, perform:
    ```
    # Put ALL of the current pip3 packages on your system in the submission
    >> pip3 freeze >> requirements.txt
    >> cat requirements.txt
    aicrowd_api
    coloredlogs
    matplotlib
    pandas
    [...]
    ```

We would suggest participants to keep the `requirements.txt` to the minimum, with only necessary packages in it. Chances are that, the more (unnecessary) packages you put in it, the more likely you may encounter an error on some (maybe totally unnecessary) packages. 

* `apt.txt` -- The Debian packages (via aptitude) used by your inference code!

These files are used to construct your **AIcrowd submission docker containers** in which your code will run.

* `Dockerfile` -- `Dockerfile` gives you more flexibility on defining the software runtime used during evaluations. The `Dockerfile` under the root path of the starter kit will be used to build your solution. Feel free to modify anything in it, and test it locally. 

----

To test your image builds locally, you can use [repo2docker](https://github.com/jupyterhub/repo2docker)
