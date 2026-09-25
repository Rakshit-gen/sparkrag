FROM python:3.11-slim

# pyspark needs a real JVM on top of python, openjdk-17-jre-headless is the
# smallest package that gives us that on debian slim (python:3.11-slim is
# debian-based, so apt is the right package manager here).
#
# The actual install path (java-17-openjdk-amd64 vs -arm64) depends on the
# build architecture, so resolve it from `java` itself instead of hardcoding
# it, that keeps this working on both amd64 and arm64 base images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s "$(dirname $(dirname $(readlink -f $(which java))))" /opt/java

ENV JAVA_HOME=/opt/java
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

COPY sample_docs ./sample_docs

ENTRYPOINT ["python", "-m", "sparkrag.cli"]
CMD ["--help"]
