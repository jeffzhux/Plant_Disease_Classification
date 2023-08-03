var express = require('express');
const fs = require('fs')

var router = express.Router();

function getFiles(dir, files = []){
  const fileList = fs.readdirSync(dir);
  for (const file of fileList) {
    const link = `${dir}/${file}`;
    if (fs.statSync(link).isDirectory()) {
      getFiles(link, files);
    } else {
      const dir_list = dir.split('/').slice(2);
      const new_link = '../' + dir_list.join('/') + '/' + file;
      files.push({'name':dir_list[dir_list.length-1], 'link':new_link.replace()});
    }
  }
  return files;
}

/* GET home page. */
router.get('/', function(req, res, next) {
  const files = getFiles('./public/demo');
  console.log(files)
  res.render('index', { files: files });
});

module.exports = router;
