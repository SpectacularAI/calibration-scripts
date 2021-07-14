# Building and publishing Docker image

1. Update version number in `build_dockerfile.sh`
2. Build it `./build_dockerfile.sh`
3. Login to Docker content registry, see [official documentation](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), tll;dr: Create PAT token in Github developer settings, add it to your env via `.bashrc` for example `export CR_PAT=YOUR_TOKEN`, login with `echo $CR_PAT | docker login ghcr.io -u spectacularai --password-stdin`. Username is intentionally organization.
4. Push your results to registry (update version number here), THEY ARE IMMEDIATELY PUBLIC `docker push ghcr.io/spectacularai/depthai-library:1.0`
