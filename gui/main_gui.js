overlay/** DATA STRUCTURE AND FUNCTIONS **/
// flask_url = 'http://52.205.255.228:8080';       // the security is fine, unless in airport where someone sniffs your data. but at home or school, fine.
flask_url = 'http://18.82.5.195:5000';
// flask_url = 'http://localhost:5000';
master = { id: 'master', name: 'master', label: 'master', map: { nodes: [], edges: [] } } ;
cur = 'master';
clipboard = { };  // store pre-drag x|y; store cut-paste name; store path and remote_folder_name
DEBUG = 0;


var spinner = {
    start : function(){
        setTimeout(function(){
            if(!$('#spinner').is(":visible")) {
               $("#spinner, #overlay").show();
            }
        },0);
    },
    stop : function(){
         setTimeout(function(){
            //  console.log('someone stopped spinner')
             $("#spinner, #overlay").hide();
         },0 );
    }
}



// spinner.stop()

//login.
var username = 'xzhang1';
var password = '5vRPz7Ngm8rNS3Sg';


var ajaxPending = false;
$( document ).ajaxSend(function(event, xhr, settings) {
    if(!ajaxPending) {
        ajaxPending = true;
        spinner.start();
    }
    xhr.setRequestHeader("Authorization", "Basic " + btoa(username + ":" + password));
});

$( document ).ajaxStop(function() {
   // console.log('spinner stop')
   ajaxPending = false;
   spinner.stop();
});



//initial connection
//note: no response means pending forever.
function make_connection() {
    return $.get(flask_url + '/make_connection')
      .done(function(d){
        // COLORING
        $('#navbar-status').html('&#9679;');
        $('#navbar-status').css('color',d.statuscolor);
        // INITIALIZATION CONSTANTS.
        ALL_ATTR_LIST = d.ALL_ATTR_LIST;
        READABLE_ATTR_LIST = d.READABLE_ATTR_LIST;
        DEBUG = d.DEBUG;
        // // other
        // console.log('make_connection success, returned d', d)
      })
      .fail(function(xhr, status, error) {
        $('#navbar-status').text('');
      });
      // text-success: connected, text-warning: empty NODES or master, text-danger: disconnectecd
}
make_connection();

/* define clerical functions */
//copy to clipboard
function copyTextToClipboard(text) {
  var textArea = document.createElement("textarea");

  // *** This styling is an extra step which is likely not required. ***
  //
  // Why is it here? To ensure:
  // 1. the element is able to have focus and selection.
  // 2. if element was to flash render it has minimal visual impact.
  // 3. less flakyness with selection and copying which **might** occur if
  //    the textarea element is not visible.
  //
  // The likelihood is the element won't even render, not even a flash,
  // so some of these are just precautions. However in IE the element
  // is visible whilst the popup box asking the user for permission for
  // the web page to copy to the clipboard.
  //

  // Place in top-left corner of screen regardless of scroll position.
  textArea.style.position = 'fixed';
  textArea.style.top = 0;
  textArea.style.left = 0;

  // Ensure it has a small width and height. Setting to 1px / 1em
  // doesn't work as this gives a negative w/h on some browsers.
  textArea.style.width = '2em';
  textArea.style.height = '2em';

  // We don't need padding, reducing the size if it does flash render.
  textArea.style.padding = 0;

  // Clean up any borders.
  textArea.style.border = 'none';
  textArea.style.outline = 'none';
  textArea.style.boxShadow = 'none';

  // Avoid flash of white box if rendered for any reason.
  textArea.style.background = 'transparent';


  textArea.value = text;

  document.body.appendChild(textArea);

  textArea.select();

  try {
    var successful = document.execCommand('copy');
    var msg = successful ? 'successful' : 'unsuccessful';
    console.log('Copying text command was ' + msg);
  } catch (err) {
    console.log('Oops, unable to copy');
  }

  document.body.removeChild(textArea);
}


// graph related functions
function lookup(c) {
    //var c
    //global master
    return lookupi(master,c);
}

function lookupi(node, c) {
    //var node, c, i
    //use of node variable is unrecommended
    //master
    //under self
    //recursive dot path
    if (c == 'master'){
        return master;
    }
    else if (c.indexOf('.')==-1) {
        if (!('map' in node)) {
            console.log(`could not find node ${c} in ${node.name}, returned master instead. `);
            return master;
        }
        for (var i=0; i < node['map']['nodes'].length; i++) {
            if (node['map']['nodes'][i]['id'] == c) {
                return node['map']['nodes'][i]
            }
        }
        console.log(`could not find node ${c} in ${node.name}, returned master instead. `);
        return master;
    } else {
        return lookupi(lookupi(node,c.split('.')[0]), c.split('.').slice(1).join('.'))
    }
}

// function (err, data, cb): the most common format

function goto(c, option) {
    //$.get('http://google',function(data){
    // if(cb) cb(data)
    //})
    //var node, c, option
    //global cur, s
    //var cam
    //animation.plugin.animate: does not process direct movement or zooming, changes color/x/y permanenetly. not really handy.
    //multiple animation can be run at once.
    //don't use ratios larger than 10 or smaller than 0.01. TypeError: cannot read property 'hidden' of undefined might be caused.
    if (lookup(c)==master && c!='master') {
      goto('master','up');
    }
    else {
      var cam = s.camera;
      if (option=='up') {
          sigma.misc.animation.camera(cam, { x:0, y:-500 }, {duration: 500,    onComplete: function (){
              s.graph.clear();
              s.graph.read(lookup(c)['map']);
              s.refresh();
              sigma.misc.animation.camera(cam, { x:0, y:500 }, {onComplete: function (){sigma.misc.animation.camera(cam, { x:0, y:0 }, {duration:500})}});}
          });
      }
      else if (option=='down') {
          sigma.misc.animation.camera(cam, { x:0, y:500 }, {duration: 500, onComplete: function (){
              s.graph.clear();
              s.graph.read(lookup(c)['map']);
              s.refresh();
              sigma.misc.animation.camera(cam, { x:0, y:-500 }, {onComplete: function (){sigma.misc.animation.camera(cam, { x:0, y:0 }, {duration:500})}});}
          });
      }
      else if (option=='in') {
        sigma.misc.animation.camera(cam, { x: lookup(c)['read_cam0:x'], y:lookup(c)['read_cam0:y'],  ratio:0.1 }, { duration: 300, onComplete: function(){
          s.graph.clear(); s.graph.read(lookup(c)['map']); s.refresh();
          sigma.misc.animation.camera(cam, { ratio:10 }, { onComplete: function(){
            sigma.misc.animation.camera(cam, { ratio:1 }, { duration: 200 });
          } });
        } });
      }
      else if (option=='out') {
          sigma.misc.animation.camera(cam, { ratio:10 }, {duration: 200, onComplete: function (){
              s.graph.clear(); s.graph.read(lookup(c)['map']); s.refresh();
              sigma.misc.animation.camera(cam, { ratio:0.1 }, {onComplete: function (){sigma.misc.animation.camera(cam, { ratio:1 }, {duration:500})}});}
          });
      }
      else if (option=='test') {
        sigma.misc.animation.camera(cam, { x: lookup(c)['read_cam0:x'], y:lookup(c)['read_cam0:y'],  ratio:0.1 }, { duration: 200, onComplete: function(){
          s.graph.clear(); s.graph.read(lookup(c)['map']); s.refresh();
          sigma.misc.animation.camera(cam, { ratio:10 }, { onComplete: function(){
            sigma.misc.animation.camera(cam, { ratio:1 }, { duration: 200 });
          } });
        } });
      }
      else {
        s.graph.clear(); s.graph.read(lookup(c)['map']);  s.refresh();
      }
      cur = c;
      $('#cur').val(cur);
    }
}

function back() {
    //global cur
    if (cur != 'master') {
        cur = cur.split('.').slice(0,cur.split('.').length-1).join('.');
        goto(cur, 'out');
    }
}

function read(d, cb) {
    //var d
    //global s, cur, master
    // s.graph.clear();

    if(d == null) {
        console.log('read: input is null. 我是说在座的各位，都是垃圾.'); return -1;
    }
    if('error' in d) {
        console.log('gui error: '+d.error); return -1;
    }
    master = d;
    if(!('map' in master)) {
        master.map = {nodes: [], edges: []};
    }
    goto(cur, '');

    if(cb) cb();
}

function log(d) {
    console.log(d);
}

function import_markdown(){
    $.get(flask_url + '/reset_NODES', function(){
        $.get(flask_url + '/import_markdown', function(){
            $.get(flask_url + '/request_', read);
        });
    });
}
function new_(){
    $.get(flask_url + '/new_', log)
    .then(function(){
        refresh();  // then must follow a promise. either use q promise, or $.get returns a promise.
    });
}
function save(){
    $.post(flask_url + '/dump_sigma', JSON.stringify(master), function(d){  // d var is 'bad' only in the sense that your collaborators don't understand.
        $.get(flask_url + '/dump_nodes', log);
    });
}
function load(){
    $.get(flask_url + '/load_nodes', function(){


        $.get(flask_url + '/load_sigma', function(d){ read(d, refresh);});
    });
}
function ipython(){
    $.ajax({
        url: flask_url + '/ipython',
        success: log,
        async: true     // async is a mentality. you start something, show a spinner, complete something, hide spinner; the fact that it's in series doesn't mean it's synchronous. ajax is async!
    });
    console.log('hiding spinner during ipython; however, flask is still occupied and no get/post will go through.')
    spinner.stop();
}

function load_datetime(datetime){
    $.post(flask_url + '/load_nodes', JSON.stringify({'datetime':datetime}), function(d){
        $.post(flask_url + '/load_sigma', JSON.stringify({'datetime':datetime}), function(d){
            read(d, refresh)
        });
    });
}
function refresh(err, data, cb){
    $.post(flask_url + '/request_', JSON.stringify(master), function(d){
        read(d);
        if(cb && typeof cb === "function") {
            cb();
        }
    });
}

function new_node(){
    //global cur
    $.post(flask_url + '/new_node', JSON.stringify({'cur':cur, 'name':'newnode'}), refresh);
}
function del_node(err, name, cb){
    //global cur
    $.post(flask_url + '/del_node', JSON.stringify({'cur':cur, 'name':name}), function(){
        refresh(null, null, function(){
            if(cb) cb();
        });
    });

}
function reset_node(name){
    //global cur
    $.post(flask_url + '/reset_node', JSON.stringify({'cur':cur + '.' + name}), refresh);
}
function compute_node(name){
    //global cur
    //var name, path_prefix
    refresh(null, null, function(){ //necessary since path needs to be reread
        $.post(flask_url + '/compute_node', JSON.stringify({'cur':cur, 'name':name}), refresh);
    });
}
function setinterval_compute_node(name){
    $.post(flask_url + '/setinterval_compute_node', JSON.stringify({'cur':cur, 'name':name}), refresh);
}
function stop_setinterval_compute_node(){
    $.get(flask_url + '/stop_setinterval_compute_node', refresh);
}

function paste_ref(){
  $.post(flask_url + '/paste_ref', JSON.stringify({'cur':cur}), function(d){
      refresh();
  })
}
function cut_ref(name){
  //global cur, var name
  $.post(flask_url + '/cut_ref', JSON.stringify({'cur':cur, 'name':name}), function(d){
      refresh();
  });
}
function duplicate_node(name){
  $.post(flask_url + '/duplicate_node', JSON.stringify({'cur':cur, 'name':name}), refresh);
}

function edit_vars(name){
        //global cur
        //var j
        //extract all textareas on page.
        var j = {};
        $('textarea.edit_vars_textarea').each(function() {
            // this.id, this.value
            j[this.id] = this.value;
        });
        j['cur'] = cur + '.' + name;
        $.post(flask_url + '/edit_vars', JSON.stringify(j), refresh);
}
function edit_vars_addfield(){
    var key = $('input#edit_vars_addfield_input').val();
    if (key.length<2) {
      throw 'New key length smaller than 2. Wth?';
    }
    if (READABLE_ATTR_LIST.indexOf(key)==-1 || $(`.edit_vars_textarea#${key}`).length!=0) {
      throw 'Key is not illegal. not added.';
    }
    $('form#edit_vars_form').append(`
                  <div class="form-group">
                    <label for="${key}">${key}</label>
                    <textarea ${key=="map"?'readonly':''} rows=${key=="property"?6:1} class="form-control edit_vars_textarea" id="${key}"></textarea>
                  </div>`);
}
function edit_vars_delfield(name){
    var key = $('input#edit_vars_addfield_input').val();
    if ($(`.edit_vars_textarea#${key}`).length==0) {
      throw 'Field not found, thus not deleted.';
    }
    $(`.edit_vars_textarea#${key}`).closest('.form-group').remove();
    $.post(flask_url + '/del_attr', JSON.stringify({'cur': cur + '.' + name, 'attr_name': key}), log);
}

function add_edge(src, dst) {
  //var src, dst
  //global cur
  //allow double add edge mechanism
  $.post(flask_url + '/add_edge', JSON.stringify({'src':src, 'dst':dst, 'cur':cur}), refresh);
}
function del_edge(src, dst) {
  //var src, dst
  //global cur
  //allow double add edge mechanism
  $.post(flask_url + '/del_edge', JSON.stringify({'src':src, 'dst':dst, 'cur':cur}), refresh);
}

function edit_node(name){
  //global cur
  //var text
  $.post(flask_url + '/get_text', JSON.stringify({'cur':cur + '.' + name}), function(d) {
    $(".sigma-tooltip").removeClass('list-group');
    $(".sigma-tooltip").addClass('panel');
    $(".sigma-tooltip").addClass('panel-default');
    $('.sigma-tooltip').html(`
          <div class="panel-heading">
            <h3 class="panel-title">${name}</h3>
          </div>
          <div class="panel-body">
            <form>
                <div class='form-group'>
                  <textarea rows=9 id='edit_node' class='form-control'>${d.text}</textarea>
                </div>
                <button type="button" class="btn btn-default" onclick="submit_edit('${name}')"">Edit</button>
            </form>
          </div>
      `
    );
  });
}
function submit_edit(name) {
      //global cur
      //var name, text
      text = $('#edit_node').val();
      j = {'cur':cur+'.'+name, 'text':text};
      $.post(flask_url + '/edit', JSON.stringify(j), refresh);
}

function copy_cur(name) {
    copyTextToClipboard(cur + '.' + name);
}

function copy_remote_folder_name(name) {
    copyTextToClipboard(clipboard['remote_folder_name']);
    delete clipboard['remote_folder_name']
}

function copy_path(name) {
    copyTextToClipboard(clipboard['path']);
    delete clipboard['path'];
}


/* EXECUTION */

/* SIGMA.JS RENDERER */

sigma.renderers.def = sigma.renderers.canvas;

s = new sigma({
  graph: lookup(cur)['map'],
  container: 'graph-container',
  renderer: {
    container: document.getElementById('graph-container'),
    type: 'canvas'
  },
  settings : {
    //Base
    labelThreshold: 1, labelHoverShadow: 'default', labelHoverShadowColor: '#000', labelAlignment: 'top', labelSize: 'proportional', labelSizeRatio: '1', labelColor: 'node',
    defaultEdgeType: 'tapered',
    minNodeSize: 6, maxNodeSize: 12, minEdgeSize: 2, maxEdgeSize: 4, //Edit these to edit outlook. Note that x,y are unrelated in scale.
    doubleClickEnabled: false, //Do not zoom on double click
    mouseWheelEnabled: false, //Do not zoom on mouse wheel
    font: 'Courier',  //Mono font
    //Drag nodes related
    dragNodeStickiness: 0.01,
    enableEdgeHovering: true,
    //Zoom
    zoomMin: 0.001,
    zoomMax: 10,
  }
});

/* DRAGGING */
// Instanciate the ActiveState plugin:
var activeState = sigma.plugins.activeState(s);
// Initialize the dragNodes plugin:
var dragListener = sigma.plugins.dragNodes(s, s.renderers[0], activeState);
// Beware: select and keyboard plugin gives the graph-container a blue-ish glow.
// Curve parallel edges:
sigma.canvas.edges.autoCurve(s);
s.refresh();

/* TOOLTIPS*/
var config = {
  node: [{
    show: 'clickNode',
    hide: 'clickStage',
    cssClass: 'panel panel-default sigma-tooltip',//'sigma-tooltip panel panel-default udf-tooltip',
    position: 'bottom',
    // autoadjust: true,
    template: `
        <div class="panel-heading">
          <h3 class="panel-title">{{{name}}}</h3>
        </div>
        <div class="panel-body">
          {{{tmp}}}
        </div>`,
    renderer: function(node, template) {
        node.tmp = '<form id="edit_vars_form">';
        //get keys list and sort it
        var tmp_key_list = Object.keys(node);
        tmp_key_list.sort(function (a,b){
          return ALL_ATTR_LIST.indexOf(a) - ALL_ATTR_LIST.indexOf(b);
        });
        //print
        for (var i=0; i<tmp_key_list.length; i++) {
          key = tmp_key_list[i];
          if (ALL_ATTR_LIST.indexOf(key)!=-1 && key!='map') {
            var rows = node[key].split(/\r\n|\r|\n/).length;
            var cols = Math.max.apply(Math, node[key].split(/\r\n|\r|\n/).map(function (el) { return el.length })); // Yeah, I don't know either. https://stackoverflow.com/questions/6043471/get-maximum-length-of-javascript-array-element
            node.tmp += `<div class="form-group">
                              <label for="${key}">${key}</label>
                              <textarea rows=${rows} cols=${cols} class="form-control edit_vars_textarea" id="${key}">${node[key]}</textarea>
                            </div>`;  // avoid classname = funcname
          }
        }
        node.tmp += `
        </form>
        <div class='row'>
          <div class="col-md-4">
            <div class="input-group" style='min-width: 350px; max-width:450px;'>
              <span class="input-group-btn">
                <button class="btn btn-default" type="button" onclick="edit_vars_addfield()">Add</button>
                <button class="btn btn-default" type="button" onclick="edit_vars_delfield('${node.name}')">Del</button>
              </span>
              <input type="text" class="form-control" id='edit_vars_addfield_input'>
              <span class="input-group-btn">
                <button type="button" class="btn btn-default" onclick="edit_vars('${node.name}')">Edit</button>
              </span>
            </div>
          </div>
        </div>
        `;

        return Mustache.render(template, node);
  }
}, {
    show: 'rightClickNode',
    hide: 'clickStage',
    cssClass: 'list-group sigma-tooltip',
    position: 'right',
    template:
    `
    <a href="#" onclick="compute_node('{{{name}}}')" class="list-group-item list-group-item-success">compute</a>
    <a href="#" onclick="copy_cur('{{{name}}}')" class="list-group-item list-group-item-info">copy cur</a>
    <a href="#" onclick="copy_path('{{{name}}}')" class="list-group-item list-group-item-info">copy path</a>
    <a href="#" onclick="copy_remote_folder_name('{{{name}}}')" class="list-group-item list-group-item-info">copy remote path</a>
    <a href="#" onclick="cut_ref('{{{name}}}')" class="list-group-item list-group-item-warning">cut</a>
    <a href="#" onclick="duplicate_node('{{{name}}}')" class="list-group-item list-group-item-warning">duplicate</a>
    <a href="#" onclick="del_node(null, '{{{name}}}')" class="list-group-item list-group-item-danger">delete</a>
    <!--<a href="#" onclick="edit_node('{{{name}}}')" class="list-group-item list-group-item-danger">import new nodes and edit</a>-->
    <a href="#" onclick="reset_node('{{{name}}}')" class="list-group-item list-group-item-danger">reset moonphase</a>
    <a href="#" onclick="setinterval_compute_node('{{{name}}}')" class="list-group-item list-group-item-success">setinterval compute</a>
    <a href="#" onclick="stop_setinterval_compute_node()" class="list-group-item list-group-item-success">stop setinterval compute</a>
    `,
    renderer: function(node, template) {
        $.post(flask_url + '/copy_remote_folder_name', JSON.stringify({'cur':cur + '.' + node.name}), function(d) {
            clipboard['remote_folder_name'] = d['remote_folder_name'];
        });
        $.post(flask_url + '/copy_path', JSON.stringify({'cur':cur + '.' + node.name}), function(d) {
            clipboard['path'] = d['path'];
        });
        return Mustache.render(template, node);
    }
  }],
  edge: {
    show: 'rightClickEdge',
    hide: 'clickStage',
    cssClass: 'list-group sigma-tooltip',
    position: 'right',
    template:
    `<a href="#" onclick="del_edge('{{{source}}}','{{{target}}}')" class="list-group-item">delete</a>
    `,
    renderer: function(edge, template) {
      return Mustache.render(template, edge);
    }
  },
  stage: {
    show: 'rightClickStage',
    hide: 'clickStage',
    position: 'right',
    cssClass: 'list-group sigma-tooltip',
    renderer: function() {
      var result = `<a href="#" onclick="back()" class="list-group-item">back</a>
      <a href="#" onclick="refresh()" class="list-group-item">refresh</a>
      <a href="#" onclick="new_node()" class="list-group-item">new node</a>
      <a href="#" onclick="paste_ref()" class="list-group-item">paste</a>`;
      return result;
    }

  }
};
// Instanciate the tooltips plugin with a Mustache renderer for node tooltips:
var tooltips = sigma.plugins.tooltips(s, s.renderers[0], config);
// Manually open a tooltip on a node:
// var n = s.graph.nodes('n10');
// var prefix = s.renderers[0].camera.prefix;
// tooltips.open(n, config.node[0], n[prefix + 'x'], n[prefix + 'y']);



s.bind('clickStage', function(e) {
  $('.sigma-tooltip').remove();
});
// s.bind('rightClickStage', function(e) {
//   console.log('clipboard is ', clipboard);
//   console.log('but result is ','cut_name' in clipboard?'yes':'no');
// });

function dist(x1,y1,x2,y2) {
  var dx = x1-x2;
  var dy = y1-y2;
  return Math.sqrt( dx*dx + dy*dy );
}

s.bind('doubleClickNode', function(e) {
    //local e, n
    if ('map' in e.data.node) {
        goto(cur + '.' + e.data.node.name, 'in');
    }
});

//drag drop draw node
dragListener.bind('startdrag', function(event) {
  clipboard['pre_drag_x']=event.data.node.x;
  clipboard['pre_drag_y']=event.data.node.y;
});
dragListener.bind('dragend', function(event) {
  //var l, i
  //global cur
  clipboard['post_drag_x']=event.data.node.x;
  clipboard['post_drag_y']=event.data.node.y;

  var l = s.graph.nodes();
  for (var i=0; i < l.length; i++) {
    n = l[i];
    if (n.name != event.data.node.name && dist(n.x, n.y, clipboard['post_drag_x'], clipboard['post_drag_y'])<0.04) {
      //  console.log('drag event cancelled, edge drawn');
      event.data.node.x=clipboard['pre_drag_x'];
      event.data.node.y=clipboard['pre_drag_y'];
      add_edge(event.data.node.name, n.name);
    }
  }
  delete clipboard.pre_drag_x; delete clipboard.pre_drag_y; delete clipboard.post_drag_x; delete clipboard.post_drag_y;
});
