import express from "express"

const app = express();
const port = 3000;

app.get("/", (req, res)=> {
    res.json({msg:"Hello i am hand detector"});
})

app.listen(port, () => {
    console.log(`app listen on port ${port}`);
    
})