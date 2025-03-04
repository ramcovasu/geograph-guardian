from typing import Dict, Any, List
import json
import re
from src.utils.db_connection import ArangoDB
from src.llm.provider import get_llm_provider

class QueryProcessor:
    def __init__(self):
        self.llm = get_llm_provider()
        self.db = ArangoDB()
        self.graph_name = 'supplychain'
        self.schema_context = self._build_schema_context()


    def _init_query_templates(self) -> Dict[str, str]:
        """Initialize query templates with graph awareness."""
        return {
            "simple_filter": """
                FOR {var} IN {collection}
                FILTER {condition}
                LIMIT {limit}
                RETURN {var}
            """.strip(),
            
            "graph_traversal": """
                FOR {from_var} IN {from_collection}
                FOR {edge_var} IN {edge_collection}
                FILTER {edge_var}._from == {from_var}._id
                FOR {to_var} IN {to_collection}
                FILTER {edge_var}._to == {to_var}._id
                LIMIT {limit}
                RETURN {{ 
                    from: {from_var},
                    to: {to_var},
                    edge: {edge_var}
                }}
            """.strip(),
            
            "risk_analysis": """
                FOR s IN suppliers
                FOR r IN risk_factors
                FILTER s.supplier_id == r.supplier_id
                SORT r.risk_score {sort_order}
                LIMIT {limit}
                RETURN {{
                    supplier: s.supplier_name,
                    risk_score: r.risk_score,
                    risk_category: r.risk_category
                }}
            """.strip(),
            
            "part_supplier": """
                FOR p IN parts
                FOR e IN supplier_provides_part
                FILTER e._to == p._id
                FOR s IN suppliers
                FILTER e._from == s._id
                LIMIT {limit}
                RETURN {{
                    part: p.part_name,
                    supplier: s.supplier_name,
                    is_primary: e.is_primary_supplier
                }}
            """.strip()
        }

    
    def _get_schema_info(self) -> Dict:
        """Get schema information from ArangoDB."""
        try:
            schema = {
                "collections": [],
                "relationships": []
            }

            # Get all collections
            collections = self.db.db.collections()
            print("\nDEBUG: All collections:")
            for c in collections:
                if not c['name'].startswith('_'):
                    print(f"Name: {c['name']}, Type: {c['type']}")

            # Process non-system collections
            for collection in collections:
                if not collection['name'].startswith('_'):
                    try:
                        print(f"\nProcessing collection: {collection['name']}")
                        collection_info = self.db.db.collection(collection['name'])
                        print(f"Collection type: {collection_info.properties()['type']}")
                        
                        sample_docs = self._get_sample_document(collection['name'])
                        print(f"Sample doc: {sample_docs[0] if sample_docs else 'No samples'}")
                        
                        if sample_docs:
                            # Get fields from the sample document
                            fields = [k for k in sample_docs[0].keys() if not k.startswith('_')]
                            
                            if collection_info.properties()['type'] == 3:  # Edge collection
                                print(f"Found edge collection: {collection['name']}")
                                if '_from' in sample_docs[0] and '_to' in sample_docs[0]:
                                    from_coll = sample_docs[0]['_from'].split('/')[0]
                                    to_coll = sample_docs[0]['_to'].split('/')[0]
                                    print(f"Edge relationship: {from_coll} → {to_coll}")
                                    schema["relationships"].append({
                                        "name": collection['name'],
                                        "from_collection": from_coll,
                                        "to_collection": to_coll,
                                        "fields": fields
                                    })
                            else:  # Document collection
                                print(f"Found document collection: {collection['name']}")
                                schema["collections"].append({
                                    "name": collection['name'],
                                    "fields": fields
                                })
                    except Exception as e:
                        print(f"Error processing collection {collection['name']}: {str(e)}")
                        continue

            print("\nFinal Schema:")
            print("Collections:", [c['name'] for c in schema['collections']])
            print("Relationships:", [r['name'] for r in schema['relationships']])
            
            return schema
        except Exception as e:
            raise Exception(f"Error getting schema info: {str(e)}")
        
    def _build_schema_context(self) -> str:
        """Build schema context with dynamic graph relationships."""
        try:
            # Get current schema info dynamically
            schema = self._get_schema_info()
            
            # Start building context with dynamic schema information
            context = "Database Schema:\n\nCollections:\n"
            
            # Add document collections dynamically
            for collection in schema["collections"]:
                context += f"- {collection['name']}: {', '.join(collection['fields'])}\n"
            
            # Add relationships section dynamically
            context += "\nRelationships (Graph Edges):\n"
            for relationship in schema["relationships"]:
                context += f"- {relationship['name']}: {relationship['from_collection']} → {relationship['to_collection']}\n"
                if relationship['fields']:
                    context += f"  Fields: {', '.join(relationship['fields'])}\n"

            # Add static examples showcasing common query patterns
            context += """
    Common Query Examples:

    1. Supplier Risk Analysis:
    FOR s IN suppliers
        FOR r IN risk_factors
        FILTER r.supplier_id == s.supplier_id
        SORT r.risk_score DESC
        LIMIT 5
        RETURN {
            supplier: s.supplier_name,
            country: s.country,
            risk_score: r.risk_score,
            risk_category: r.risk_category
        }

    2. Critical Parts with Low Inventory:
    FOR p IN parts
        FILTER p.criticality_level == "HIGH"
        FOR i IN inventory
        FILTER i.part_id == p.part_id
        FILTER i.quantity_on_hand < i.safety_stock
        FOR sp IN supplier_provides_part
        FILTER sp._to == p._id
        FOR s IN suppliers
        FILTER s._id == sp._from
        RETURN {
            part: p.part_name,
            current_stock: i.quantity_on_hand,
            safety_stock: i.safety_stock,
            supplier: s.supplier_name,
            lead_time: sp.lead_time_days
        }

    3. Part Dependency Chain Analysis:
    FOR p IN parts
        FOR d IN 1..3 OUTBOUND p part_depends_on
        LET suppliers = (
            FOR sp IN supplier_provides_part
            FILTER sp._to == d._id
            FOR s IN suppliers
            FILTER s._id == sp._from
            RETURN {
                supplier_name: s.supplier_name,
                lead_time: sp.lead_time_days
            }
        )
        RETURN {
            part: p.part_name,
            dependent_part: d.part_name,
            suppliers: suppliers
        }

    4. Purchase Order Analysis with Supplier Risk:
    FOR po IN purchase_orders
        FILTER po.status == 'PENDING'
        FOR s IN suppliers
        FILTER s.supplier_id == po.supplier_id
        FOR r IN risk_factors
        FILTER r.supplier_id == s.supplier_id
        SORT r.risk_score DESC
        LIMIT 10
        RETURN {
            order_id: po.po_id,
            supplier: s.supplier_name,
            risk_score: r.risk_score,
            order_date: po.order_date,
            status: po.status
        }

    5. Inventory Transactions by Part:
    FOR t IN inventory_transactions
        FILTER t.transaction_type == 'STOCKOUT'
        FOR p IN parts
        FILTER p.part_id == t.part_id
        FOR sp IN supplier_provides_part
        FILTER sp._to == p._id
        FOR s IN suppliers
        FILTER s._id == sp._from
        SORT t.transaction_date DESC
        LIMIT 5
        RETURN {
            part: p.part_name,
            transaction_type: t.transaction_type,
            quantity: t.quantity,
            date: t.transaction_date,
            supplier: s.supplier_name
        }

    6. What are the dependent parts for LIDAR-X1 and their suppliers:
    FOR p IN parts
    FILTER p.part_name == "LIDAR-X1"
    FOR pd IN part_depends_on 
        FILTER pd._from == p._id
        FOR dp IN parts
            FILTER dp._id == pd._to
            LET part_suppliers = (
                FOR sp IN supplier_provides_part
                FILTER sp._to == dp._id
                FOR s IN suppliers
                FILTER s._id == sp._from
                RETURN {
                    name: s.supplier_name,
                    country: s.country,
                    rating: s.supplier_rating
                }
            )
            RETURN {
                part: p.part_name,
                dependent_part: dp.part_name,
                suppliers: part_suppliers
            }

    Rules for Query Generation:
    - Use actual collection names (suppliers, parts, inventory, etc.)
    - When returning object data, specify exact fields rather than whole objects
    - Include SORT for meaningful ordering of results
    - Use LIMIT to restrict result size
    - Leverage graph traversal with OUTBOUND for relationship queries
    - Use subqueries (LET) for complex relationship chains
    - Include relevant fields in the output
    - Join collections through defined edge collections 
    - Filter conditions should match field names exactly

    Query Patterns:
    - For graph traversals: FOR v IN 1..n OUTBOUND start_vertex edge_collection
    - For multiple filters: Chain FILTER conditions
    - For aggregations: Use COLLECT and AGGREGATE
    - For complex joins: Combine FOR loops with FILTER conditions
    - For nested data: Use subqueries with LET
 
    """
            
            return context

        except Exception as e:
            self.logger.error(f"Error building schema context: {str(e)}")
            raise Exception(f"Error building schema context: {str(e)}")    
            
    def _get_sample_document(self, collection_name: str) -> List[Dict]:
        """Get a sample document from a collection, ensuring edges include _from and _to."""
        query = f"FOR doc IN {collection_name} LIMIT 1 RETURN doc"
        print(f"Executing Query: {query}")  # Debug
        cursor = self.db.db.aql.execute(query)
        docs = list(cursor)
        print(f"Sample Document from {collection_name}: {docs}")  # Debug
        return docs


    def generate_aql(self, question: str) -> str:
        """Generate and validate AQL query."""
        try:
            response = self.llm.generate_response(self.schema_context, question)
            clean_query = self._clean_and_validate_query(response)
            print(f'generated query is {clean_query}')
            return clean_query
        except Exception as e:
            raise Exception(f"Error generating AQL: {str(e)}")

    def _clean_and_validate_query(self, query: str) -> str:
        """Clean and validate AQL query."""
        query = query.strip()
        if query.startswith("```"):
            query = query.split("```")[1]
            if query.startswith("aql"):
                query = query[3:]
        query = query.strip("` \n")
        
        # Basic structure validation
        if not query.upper().startswith("FOR"):
            raise ValueError("Invalid AQL: Query must start with FOR")
            
        if "RETURN" not in query.upper():
            raise ValueError("Invalid AQL: Query must include RETURN statement")
        
        # Validate collection names
        collections = self.db.db.collections()
        collection_names = [c['name'] for c in collections if not c['name'].startswith('_')]
        
        for word in query.split():
            if word.lower() == "in":
                collection = query.split(word)[1].split()[0]
                if collection not in collection_names:
                    raise ValueError(f"Invalid collection name: {collection}")
        
        # Add LIMIT if not present
        if "LIMIT" not in query.upper():
            parts = query.split("RETURN")
            query = f"{parts[0]} LIMIT 100 RETURN{parts[1]}"
            
        return query

    def execute_query(self, aql_query: str) -> List[Dict[str, Any]]:
        """Execute AQL query with error handling."""
        try:
            cursor = self.db.db.aql.execute(aql_query)
            return list(cursor)
        except Exception as e:
            raise Exception(f"Error executing AQL query: {str(e)}")
        
    def explain_results(self, results: List[Dict[str, Any]], query: str) -> str:
        return self.llm.explain_results(results, query)
   

    def suggest_visualization(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return self.llm.suggest_visualization(results)
        except Exception:
            return self._default_visualization()