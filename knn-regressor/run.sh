#Only for local test runs. Please define env variables accordingly
echo ""
echo ""
echo ""
echo "This is a test run"
echo ""
echo ""
echo ""
sleep 4

./shut-down.sh
docker build -t scikit_multiflow_1 .
docker run -d --name scikit_multiflow_1 scikit_multiflow_1
docker logs scikit_multiflow_1 --follow