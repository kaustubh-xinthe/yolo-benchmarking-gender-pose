@classmethod
    def __build_roi_name_filter(cls, roi_name_list: List[str]) -> str:

        roi_name_conditions = " OR ".join([f"roi.name ILIKE '%%{rname}%%'" for rname in roi_name_list])

        subquery = f"""
            WITH roi_matching AS (
                SELECT roi.unique_id, roi.name
                FROM public.regionofinterest roi
                WHERE {roi_name_conditions}
            )
            SELECT ed.identifier
            FROM public.eventdefinition ed
            JOIN roi_matching rm ON (
                EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(ed.definition->'detections') AS detections,
                         jsonb_array_elements(detections->'regions_of_interest') AS regions
                    WHERE regions->>'roi_id' = rm.unique_id::text
                ) OR EXISTS (
                    SELECT 1
                    FROM jsonb_array_elements(ed.definition->'deactivation'->'detections') AS deactivations,
                         jsonb_array_elements(deactivations->'regions_of_interest') AS regions
                    WHERE regions->>'roi_id' = rm.unique_id::text
                )
            )
            WHERE ed.is_active is true
            GROUP BY ed.identifier
        """

        return f"ed.identifier IN ({subquery})"
    # query to search events using roi names

    @classmethod
    def __create_filter_clause(cls,
                               eventdefinition_id: Optional[str] = None,
                               event_name: Optional[str] = None,
                               camera_id: Optional[str] = None,
                               camera_name: Optional[str] = None,
                               roi_id: Optional[str] = None,
                               roi_name: Optional[str] = None,
                               from_date: Optional[datetime] = None,
                               to_date: Optional[datetime] = None,
                               from_time: Optional[time] = None,
                               to_time: Optional[time] = None) -> str:
        try:
            conditions = []

            if eventdefinition_id:
                eventdefinition_list = [UUID(ed_id.strip()) for ed_id in eventdefinition_id.split(',')]
                eventdefinition_conditions = " OR ".join(
                    [f"ed.identifier = '{eventdefinition_id}'::uuid" for eventdefinition_id in eventdefinition_list]
                )
                conditions.append(f"({eventdefinition_conditions})")

            if camera_id:
                camera_list = [int(cam_id.strip()) for cam_id in camera_id.split(',')]
                camera_conditions = " OR ".join([f"ev.camera_id = {camera_id}" for camera_id in camera_list])
                conditions.append(f"({camera_conditions})")

            if roi_id:
                roi_list = [UUID(r_id.strip()) for r_id in roi_id.split(',')]
                roi_conditions = " OR ".join(
                    [
                        f"(ed.definition->'detections' @> '[{{\"regions_of_interest\": [{{\"roi_id\": \"{roi_id}\"}}]}}]' OR " +
                        f"ed.definition->'deactivation'->'detections' @> '[{{\"regions_of_interest\": [{{\"roi_id\": \"{roi_id}\"}}]}}]')"
                        for roi_id in roi_list]
                )
                conditions.append(f"({roi_conditions})")

            if event_name:
                event_name_list = [ename.strip() for ename in event_name.split(',')]
                event_name_conditions = " OR ".join([f"ed.event_name ILIKE '%%{ename}%%'" for ename in event_name_list])
                conditions.append(f"({event_name_conditions})")

            if camera_name:
                camera_name_list = [cname.strip() for cname in camera_name.split(',')]
                camera_name_conditions = " OR ".join([f"c.name ILIKE '%%{cname}%%'" for cname in camera_name_list])
                conditions.append(f"ev.camera_id IN (SELECT id FROM public.camera c WHERE {camera_name_conditions})")

            if roi_name:
                roi_name_list = [rname.strip() for rname in roi_name.split(',')]
                roi_name_filter = EventManager.__build_roi_name_filter(roi_name_list)
                conditions.append(roi_name_filter)

            if from_date:
                conditions.append(f"ev.start_time >= '{from_date}'")

            if to_date:
                conditions.append(f"ev.start_time <= '{to_date}'")

            if from_time:
                conditions.append(f"CAST(ev.start_time AS TIME) >= '{from_time}'")

            if to_time:
                conditions.append(f"CAST(ev.start_time AS TIME) <= '{to_time}'")

            filter_clause = " AND ".join(conditions)
            if filter_clause:
                filter_clause = "(" + filter_clause + ")"

            return filter_clause
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to construct filter clause. Error: {str(e)}")
    # creating filter clause based on filter criteria - Kaustubh

    def filter_events_for_client(self, client_id: int, page: int,
                                 page_size: int,
                                 eventdefinition_id: Optional[UUID] = None,
                                 event_name: Optional[str] = None,
                                 camera_id: Optional[int] = None,
                                 camera_name: Optional[str] = None,
                                 roi_id: Optional[UUID] = None,
                                 roi_name: Optional[str] = None,
                                 from_date: Optional[datetime] = None,
                                 to_date: Optional[datetime] = None,
                                 from_time: Optional[time] = None,
                                 to_time: Optional[time] = None,
                                 sort_fields: Optional[List[EventListSortField]] = None,
                                 sort_orders: Optional[List[SortOrder]] = None) -> EventListModel:

        filter_clause = EventManager.__create_filter_clause(eventdefinition_id=eventdefinition_id,
                                                            event_name=event_name,
                                                            camera_id=camera_id,
                                                            camera_name=camera_name,
                                                            roi_id=roi_id,
                                                            roi_name=roi_name,
                                                            from_date=from_date,
                                                            to_date=to_date,
                                                            from_time=from_time,
                                                            to_time=to_time)

        if not sort_fields:
            sort_fields = []
        sort_criteria = list(zip(sort_fields, sort_orders or []))

        # Append any missing sort_fields and their default sort orders
        for field, order in DEFAULT_SORT_ORDER.items():
            if field not in [sort_field for sort_field, _ in sort_criteria]:
                sort_criteria.append((field, order))
        sort_clause = EventManager.__create_sort_clause(sort_criteria)

        list_info, events = self._datamanager.fetch_events_for_client(client_id=client_id, page=page,
                                                                      page_size=page_size,
                                                                      sort_clause=sort_clause,
                                                                      filter_clause=filter_clause)
        event_models = [EventManager.__convert_to_event_model(event) for event in events]
        return EventListModel(events=event_models,
                              page_number=list_info.page_number,
                              page_size=list_info.page_size,
                              total_count=list_info.total_count,
                              total_pages=list_info.total_pages
                              )
