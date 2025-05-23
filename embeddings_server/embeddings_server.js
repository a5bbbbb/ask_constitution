const app = require("./app");
"use strict";

const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 6660;


app
  .listen(PORT, "0.0.0.0", function () {
    console.log(`Server is running on port ${PORT}.`);
  })
  .on("error", (err) => {
    if (err.code === "EADDRINUSE") {
      console.log("Error: address already in use");
    } else {
      console.log(err);
    }
  });