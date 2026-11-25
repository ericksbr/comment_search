# Putting the comment search on the web: options

This note describes ways to offer the comment search to the public on a website, what each option needs, and what to ask the agency's web and IT staff. Today the search runs only on one person's computer, and only for staff.

## What the public would get

A search box on the rulemaking page. A visitor types words or a phrase, or a description of an idea, and sees the matching comments with the matching passage shaded. Commenter names and contact details would need a decision first (see "Privacy" below).

## Options

### A. Staff only (today)

The search runs on a staff member's computer, and the comment files stay on the agency's drive. Nothing is public.

- **Needs:** nothing new.
- **Good for:** analysts working through the record.

### B. A small search server, shown on the agency page through an embedded frame

The search runs on a small server that the agency hosts or approves, and the agency page shows it in an embedded frame (an "iframe"). The visitor sees the search inside the page, but it is running on the server.

- **Needs:** a server the agency can run (or an approved cloud service); IT sign-off; a review for accessibility (screen readers and keyboard use); a privacy review; someone to keep it running and update it.
- **Gives:** the full search, including meaning-based matching.
- **Works if:** the agency's web system allows embedded frames.

### C. A static page that runs in the visitor's browser

The page is only files: no server. The comments are prepared ahead of time and the search runs in the visitor's own browser.

- **Needs:** the web system can host plain files; no server.
- **Gives:** exact-word search with no server. Meaning-based search is harder: the language model would be downloaded into the visitor's browser (a large download), or the meaning matches would be prepared in advance for fixed topics.
- **Good for:** a low-cost, low-maintenance first public version if exact-word search is enough.

### D. A third-party hosting service

A service such as Hugging Face Spaces runs the same search for us. Fastest to try.

- **Needs:** approval to put the comment text on an outside service, and a decision on whether the public record can sit there.
- **Not recommended** for the public version unless the agency approves it in writing.

### E. Publish the comments in a searchable form instead of a search tool

Publish one searchable spreadsheet of comments, or text-searchable PDFs, or an index page with links. The agency's existing PDFs are scanned or "reduced" files, so this needs a text-extraction step first.

- **Needs:** the most preparation by staff, the least software.
- **Gives:** the public can use Ctrl+F or a spreadsheet filter. No meaning-based search.

## Several pages, several comment sets

If the agency later wants different pages with different search boxes, each pointing at a different set of comments, the design can handle that without new code for each page:

- One search engine serves all the comment sets. Each set is a folder with its comments and its saved index, plus a small settings file (title, which columns to show, default cutoffs).
- Each page embeds the same search box and names its comment set in the address, for example `search?set=div512`. A new rulemaking means adding a folder and a settings file.
- One server can hold several sets, or each set can run on its own server if the agency wants them kept apart.

## Privacy

The comments are a public record, but they often include names, addresses, and phone numbers. Before anything goes public, the agency should decide:

- Whether to show commenter names, or only comment IDs.
- Whether to remove personal details from the text.
- How long the search keeps any copy of the comments, and where.

## Questions for the agency's web and IT staff

1. What web system does the agency use for this page, and can it embed a tool from another address (an "iframe") or add custom code?
2. Who would host a small search server, and is there an approved place for one?
3. What accessibility standard applies to public web tools?
4. Is there a policy on putting public records on outside services?
5. Who maintains the tool after launch: IT, the program team, or us?
6. Does the agency want commenter names visible in the search?

## Our recommendation

1. Keep building the staff version now. Its features (picking passages, export, collapsible list) are what analysts need either way.
2. Keep the search engine separate from the window that staff use, so the same engine can serve a public page later.
3. Ask the agency's web and IT staff the questions above before building anything public. Their answers choose between B, C, and E.
4. Start with a short, approved public version (option C or E) if the answers rule out a server. Add meaning-based search once a server is approved.
