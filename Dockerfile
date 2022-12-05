# Create builder and execute builder
FROM alpine:3.16.2 AS builder

RUN apk add --no-cache bash
RUN apk add --no-cache \
    g++ \
    make

COPY Makefile .
COPY ./src ./src

RUN make all

# Create run Container
FROM alpine:3.16.2

RUN apk add --no-cache \
    bash \
    libstdc++

COPY docker-run.sh .
COPY --from=builder /dist/. /dist/.

ENTRYPOINT [ "./docker-run.sh" ]