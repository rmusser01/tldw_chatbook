"""Stock Notes folder and approved file-import effects for the elected writer."""

from functools import cache

from ..Notes.note_folder_models import FolderMutationResult, NoteFolder, NoteFolderMembership
from ..Notes.note_folder_repository import LocalNoteFolderRepository, _normalize_desired_memberships, _normalize_owner_ids
from ..Notes.note_import_executor import LocalNoteImportTarget, LocalTargetFolder, LocalTargetNote, _copy_segments, _normalize_keywords, _translate_exception
from ..Notes.note_import_plan_models import ParsedNotePayload


@cache
def value_types():
    classes = (FolderMutationResult, NoteFolder, NoteFolderMembership, LocalTargetFolder, LocalTargetNote, ParsedNotePayload)
    return {cls.__name__: cls for cls in classes}


def source_database(kind, owner):
    if kind == "folder":
        return owner.db if type(owner) is LocalNoteFolderRepository else None
    if type(owner) is LocalNoteImportTarget and type(owner._folders) is LocalNoteFolderRepository and owner._folders.db is owner._db:
        return owner._db
    return None


def prepare_iterables(kind, parameters):
    """Consume each one-shot input once using the repository's original bounds."""
    if kind == "folder":
        if "desired" in parameters:
            parameters["desired"] = tuple(_normalize_desired_memberships(parameters["desired"]))
        if "active_owner_ids" in parameters:
            parameters["active_owner_ids"] = _normalize_owner_ids(parameters["active_owner_ids"])
        return
    try:
        if "segments" in parameters:
            parameters["segments"] = _copy_segments(parameters["segments"])
        if "keywords" in parameters:
            parameters["keywords"] = tuple(_normalize_keywords(parameters["keywords"]).values())
    except Exception as exc:
        raise _translate_exception(exc) from None


def execute_notes_operation(kind, function, db, parameters):
    folders = LocalNoteFolderRepository(db)
    if kind == "folder":
        return function(folders, **parameters)
    return function(LocalNoteImportTarget(db=db, folder_repository=folders), **parameters)
