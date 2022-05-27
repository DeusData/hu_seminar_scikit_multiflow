echo "Shutting down container & cleaning up: "
docker rm -f scikit_multiflow_1
screen -X -S loggingSession quit
screen -wipe