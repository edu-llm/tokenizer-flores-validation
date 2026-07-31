/**
 * Export every hyperlink from the active Google Doc (or a fixed Doc ID).
 *
 * Setup (bound to the P2 Languages doc — recommended):
 *   1. Open https://docs.google.com/document/d/1TsCcLb4rYHy7JVVnFptp2pNTExJ0FjXFqVI5x416nxE/edit
 *   2. Extensions → Apps Script
 *   3. Paste this file, save
 *   4. Run exportLinksToDriveFile  (authorize when prompted)
 *   5. Check My Drive for "P2_Languages_links.md" / ".json" / ".csv"
 *
 * Or run exportLinksToSheet for a spreadsheet.
 */

var DOC_ID = '1TsCcLb4rYHy7JVVnFptp2pNTExJ0FjXFqVI5x416nxE';
var OUTPUT_BASENAME = 'P2_Languages_links';

/**
 * Primary entry: writes Markdown + JSON + CSV to Drive (same folder as the Doc if possible).
 */
function exportLinksToDriveFile() {
  var links = collectLinks_(openDoc_());
  var md = toMarkdown_(links);
  var json = JSON.stringify({schema_version: 1, kind: 'gdoc_link_inventory', document_id: DOC_ID, count: links.length, links: links}, null, 2);
  var csv = toCsv_(links);

  var folder = driveFolderForDoc_(DOC_ID);
  writeTextFile_(folder, OUTPUT_BASENAME + '.md', md, MimeType.PLAIN_TEXT);
  writeTextFile_(folder, OUTPUT_BASENAME + '.json', json, MimeType.PLAIN_TEXT);
  writeTextFile_(folder, OUTPUT_BASENAME + '.csv', csv, MimeType.CSV);

  Logger.log('Exported %s links to Drive folder %s', links.length, folder.getName());
  Logger.log('Markdown file: %s', OUTPUT_BASENAME + '.md');
}

/** Spreadsheet export (one row per link). */
function exportLinksToSheet() {
  var links = collectLinks_(openDoc_());
  var ss = SpreadsheetApp.create(OUTPUT_BASENAME);
  var sheet = ss.getActiveSheet();
  sheet.clear();
  sheet.appendRow(['section_path', 'anchor_text', 'url', 'element_type', 'list_id', 'paragraph_index']);
  links.forEach(function (row) {
    sheet.appendRow([
      row.section_path,
      row.anchor_text,
      row.url,
      row.element_type,
      row.list_id,
      row.paragraph_index,
    ]);
  });
  Logger.log('Spreadsheet URL: %s', ss.getUrl());
  Logger.log('Exported %s links', links.length);
}

/** Log-only dry run. */
function previewLinks() {
  var links = collectLinks_(openDoc_());
  Logger.log('count=%s', links.length);
  links.slice(0, 30).forEach(function (row, i) {
    Logger.log('%s. [%s] %s -> %s', i + 1, row.section_path, row.anchor_text, row.url);
  });
}

function openDoc_() {
  try {
    return DocumentApp.getActiveDocument() || DocumentApp.openById(DOC_ID);
  } catch (e) {
    return DocumentApp.openById(DOC_ID);
  }
}

/**
 * Walk body paragraphs; track heading path for section context.
 * Captures URL + display text for every LinkUrl on a text run.
 */
function collectLinks_(doc) {
  var body = doc.getBody();
  var n = body.getNumChildren();
  var sectionStack = [];
  var out = [];
  var seen = {};

  for (var i = 0; i < n; i++) {
    var child = body.getChild(i);
    var type = child.getType();

    if (type === DocumentApp.ElementType.PARAGRAPH) {
      var para = child.asParagraph();
      var heading = para.getHeading();
      var text = safeText_(para);
      if (heading !== DocumentApp.ParagraphHeading.NORMAL && text) {
        var level = headingLevel_(heading);
        while (sectionStack.length && sectionStack[sectionStack.length - 1].level >= level) {
          sectionStack.pop();
        }
        sectionStack.push({level: level, title: text});
      }
      extractFromTextContainer_(para, 'paragraph', i, sectionPath_(sectionStack), null, out, seen);
    } else if (type === DocumentApp.ElementType.LIST_ITEM) {
      var item = child.asListItem();
      extractFromTextContainer_(
        item,
        'list_item',
        i,
        sectionPath_(sectionStack),
        String(item.getListId()),
        out,
        seen
      );
    } else if (type === DocumentApp.ElementType.TABLE) {
      var table = child.asTable();
      for (var r = 0; r < table.getNumRows(); r++) {
        var row = table.getRow(r);
        for (var c = 0; c < row.getNumCells(); c++) {
          extractFromTextContainer_(
            row.getCell(c),
            'table_cell',
            i,
            sectionPath_(sectionStack),
            null,
            out,
            seen
          );
        }
      }
    }
  }
  return out;
}

function extractFromTextContainer_(el, elementType, paragraphIndex, sectionPath, listId, out, seen) {
  var textObj;
  try {
    textObj = el.editAsText();
  } catch (e) {
    return;
  }
  if (!textObj) return;

  var full = textObj.getText();
  if (!full) return;

  // Scan each character for link-url changes (Apps Script LinkUrl API is run-based).
  var i = 0;
  while (i < full.length) {
    var url = textObj.getLinkUrl(i);
    if (!url) {
      i++;
      continue;
    }
    var start = i;
    while (i < full.length && textObj.getLinkUrl(i) === url) {
      i++;
    }
    var anchor = full.substring(start, i).replace(/\s+/g, ' ').trim();
    if (!anchor) continue;

    var key = url + '\t' + anchor + '\t' + sectionPath;
    if (seen[key]) continue;
    seen[key] = true;

    out.push({
      section_path: sectionPath,
      anchor_text: anchor,
      url: url,
      element_type: elementType,
      list_id: listId || '',
      paragraph_index: paragraphIndex,
    });
  }
}

function headingLevel_(heading) {
  switch (heading) {
    case DocumentApp.ParagraphHeading.TITLE:
      return 0;
    case DocumentApp.ParagraphHeading.SUBTITLE:
      return 1;
    case DocumentApp.ParagraphHeading.HEADING1:
      return 1;
    case DocumentApp.ParagraphHeading.HEADING2:
      return 2;
    case DocumentApp.ParagraphHeading.HEADING3:
      return 3;
    case DocumentApp.ParagraphHeading.HEADING4:
      return 4;
    case DocumentApp.ParagraphHeading.HEADING5:
      return 5;
    case DocumentApp.ParagraphHeading.HEADING6:
      return 6;
    default:
      return 9;
  }
}

function sectionPath_(stack) {
  return stack.map(function (s) { return s.title; }).join(' > ');
}

function safeText_(el) {
  try {
    return (el.getText() || '').replace(/\s+/g, ' ').trim();
  } catch (e) {
    return '';
  }
}

function toMarkdown_(links) {
  var lines = [];
  lines.push('# P2 Languages — link inventory');
  lines.push('');
  lines.push('Source doc: https://docs.google.com/document/d/' + DOC_ID + '/edit');
  lines.push('Exported: ' + new Date().toISOString());
  lines.push('Link count: ' + links.length);
  lines.push('');

  var current = null;
  links.forEach(function (row) {
    if (row.section_path !== current) {
      current = row.section_path;
      lines.push('');
      lines.push('## ' + (current || '(no heading)'));
      lines.push('');
    }
    lines.push('- [' + escapeMd_(row.anchor_text) + '](' + row.url + ')');
  });
  lines.push('');
  return lines.join('\n');
}

function toCsv_(links) {
  var rows = [['section_path', 'anchor_text', 'url', 'element_type', 'list_id', 'paragraph_index']];
  links.forEach(function (row) {
    rows.push([
      row.section_path,
      row.anchor_text,
      row.url,
      row.element_type,
      row.list_id,
      String(row.paragraph_index),
    ]);
  });
  return rows.map(function (cols) {
    return cols.map(csvEscape_).join(',');
  }).join('\n') + '\n';
}

function csvEscape_(value) {
  var s = String(value == null ? '' : value);
  if (/[",\n\r]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function escapeMd_(s) {
  return String(s).replace(/[\[\]]/g, '\\$&');
}

function driveFolderForDoc_(docId) {
  try {
    var file = DriveApp.getFileById(docId);
    var parents = file.getParents();
    if (parents.hasNext()) {
      return parents.next();
    }
  } catch (e) {
    // Fall through to root.
  }
  return DriveApp.getRootFolder();
}

function writeTextFile_(folder, name, content, mime) {
  var existing = folder.getFilesByName(name);
  if (existing.hasNext()) {
    existing.next().setContent(content);
  } else {
    folder.createFile(name, content, mime);
  }
}
