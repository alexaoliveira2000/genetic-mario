const path = require("path")
const express = require('express');
const app = express();

app.set("hostname", "localhost");
app.set("port", 3000);
app.use(express.static(__dirname));

app.listen(app.get("port"), () => {
    console.log(`\n Server Listening on ${app.get("port")}`)
});

app.get("/", function (req, res) {
    res.sendFile(path.join(__dirname, "index.html"))
})